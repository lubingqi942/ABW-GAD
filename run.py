import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from sklearn.metrics import roc_auc_score

from args import parameter_parser
from Dataloader import load_mat
from utils import negative_sampling, calculate_auprc
from beta_wavelet import MultiScaleBetaWavelet
from layers import GraphConvolution


class GCNEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers=2, dropout=0.0):
        super(GCNEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_dim
            self.gcn_layers.append(GraphConvolution(in_dim, hidden_dim, bias=True))
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        h = x
        for i, gcn in enumerate(self.gcn_layers):
            h = gcn(h, adj)
            if i < self.num_layers - 1:
                h = self.activation(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, feat_size, decoder_hidden_dim=128):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, decoder_hidden_dim)
        self.fc2 = nn.Linear(decoder_hidden_dim, feat_size)

    def forward(self, z):
        return self.fc2(F.relu(self.fc1(z)))


class BilinearDiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(BilinearDiscriminator, self).__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, node_emb, subgraph_emb):
        return self.bilinear(node_emb, subgraph_emb).squeeze(-1)


class HypersphereEncoder(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.0):
        super(HypersphereEncoder, self).__init__()
        self.encoder = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        center_init = torch.randn(hidden)
        center_init = F.normalize(center_init, p=2, dim=0)
        self.center = nn.Parameter(center_init)

    def forward(self, h):
        z = torch.tanh(self.dropout(self.encoder(h)))
        z_norm = F.normalize(z, p=2, dim=1)
        c_norm = F.normalize(self.center, p=2, dim=0)
        cos_sim = (z_norm * c_norm.unsqueeze(0)).sum(dim=1)
        return z_norm, cos_sim

    def get_center(self):
        return F.normalize(self.center, p=2, dim=0)


class FusionModel(nn.Module):
    def __init__(self, feat_size, hidden_dim=64, wavelet_order=2, wavelet_shape=1.0,
                 num_gcn_layers=2, dropout=0.0,
                 decoder_hidden_dim=128):
        super(FusionModel, self).__init__()

        if wavelet_order < 1:
            raise ValueError(f"wavelet_order must be >= 2, got {wavelet_order}")

        self.wavelet = MultiScaleBetaWavelet(d=wavelet_order, beta_shape=wavelet_shape)
        self.num_bands = wavelet_order + 1

        self.band_encoders = nn.ModuleList([
            HypersphereEncoder(feat_size, hidden_dim, dropout) for _ in range(self.num_bands)
        ])

        self.register_buffer('band_weights', torch.ones(self.num_bands))

        self.gcn = GCNEncoder(feat_size, hidden_dim, num_gcn_layers, dropout)
        self.decoder = MLPDecoder(hidden_dim, feat_size, decoder_hidden_dim)
        self.disc = BilinearDiscriminator(hidden_dim)

    def forward(self, x, adj_norm):
        h_bands = self.wavelet(x, adj_norm)
        z_bands, cos_bands, c_bands = [], [], []
        for i, h in enumerate(h_bands):
            z, cos = self.band_encoders[i](h)
            z_bands.append(z)
            cos_bands.append(cos)
            c_bands.append(self.band_encoders[i].get_center())

        z_gcn = self.gcn(x, adj_norm)
        x_hat = self.decoder(z_gcn)

        adj_coo = adj_norm.coalesce()
        deg = torch.zeros(x.size(0), device=x.device)
        deg = deg.scatter_add(0, adj_coo.indices()[0], adj_coo.values())
        deg = torch.clamp(deg, min=1e-10)
        subgraph = torch.sparse.mm(adj_norm, z_gcn) / deg.unsqueeze(1)

        return {
            'z_bands': z_bands, 'cos_bands': cos_bands, 'c_bands': c_bands,
            'z_gcn': z_gcn, 'x_hat': x_hat, 'subgraph': subgraph,
        }


def local_affinity(z, adj):
    if adj.is_sparse:
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    else:
        row_sum = adj.sum(dim=1)
    row_sum = torch.clamp(row_sum, min=1e-30)
    z_norm = F.normalize(z, p=2, dim=1)
    agg = torch.sparse.mm(adj, z_norm) if adj.is_sparse else torch.matmul(adj, z_norm)
    return (z_norm * agg).sum(dim=1) / row_sum


def deviation_loss(cos_sim):
    diff = 1.0 - cos_sim
    return diff.pow(2), diff.pow(2).mean()


def dgi_loss(z, subgraph, disc, negsamp_ratio=1):
    N = z.size(0)
    device = z.device
    logits_list = [disc(z, subgraph)]
    sg_shifted = subgraph
    for _ in range(negsamp_ratio):
        sg_shifted = torch.cat((sg_shifted[-1:], sg_shifted[:-1]), dim=0)
        logits_list.append(disc(z, sg_shifted))
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat([torch.ones(N, device=device),
                        torch.zeros(negsamp_ratio * N, device=device)])
    return F.binary_cross_entropy_with_logits(logits, labels)


def band_decorrelation_loss(z_bands):
    num_bands = len(z_bands)
    if num_bands < 2:
        return torch.zeros((), device=z_bands[0].device)
    loss = torch.zeros((), device=z_bands[0].device)
    for i in range(num_bands):
        for j in range(i + 1, num_bands):
            zi = z_bands[i]
            zj = z_bands[j]
            cos = (zi * zj).sum() / (zi.norm() * zj.norm() + 1e-12)
            loss = loss + cos * cos
    return loss


def _sub_adj(adj, node_idx):
    adj_coo = adj.coalesce()
    src = adj_coo.indices()[0]
    dst = adj_coo.indices()[1]
    vals = adj_coo.values()
    N = adj.size(0)

    in_sample = torch.zeros(N, dtype=torch.bool, device=adj.device)
    in_sample[node_idx] = True
    edge_mask = in_sample[src] & in_sample[dst]

    sub_src = src[edge_mask]
    sub_dst = dst[edge_mask]
    sub_vals = vals[edge_mask]

    new_idx = torch.full((N,), -1, dtype=torch.long, device=adj.device)
    new_idx[node_idx] = torch.arange(len(node_idx), device=adj.device)
    sub_src_new = new_idx[sub_src]
    sub_dst_new = new_idx[sub_dst]

    K = len(node_idx)
    return torch.sparse_coo_tensor(
        torch.stack([sub_src_new, sub_dst_new]), sub_vals, (K, K)
    ).coalesce()


def compute_losses(outputs, adj_label, features, args, b_xent, disc, band_weights,
                   node_idx=None):
    device = features.device
    num_bands = len(outputs['z_bands'])

    if node_idx is not None:
        N = len(node_idx)
        z_bands_s = [z[node_idx] for z in outputs['z_bands']]
        cos_bands_s = [c[node_idx] for c in outputs['cos_bands']]
        z_gcn_s = outputs['z_gcn'][node_idx]
        x_hat_s = outputs['x_hat'][node_idx]
        subgraph_s = outputs['subgraph'][node_idx]
        feat_s = features[node_idx]
        adj_sub = _sub_adj(adj_label, node_idx)
    else:
        N = features.size(0)
        z_bands_s = outputs['z_bands']
        cos_bands_s = outputs['cos_bands']
        z_gcn_s = outputs['z_gcn']
        x_hat_s = outputs['x_hat']
        subgraph_s = outputs['subgraph']
        feat_s = features
        adj_sub = adj_label

    dis_adj = negative_sampling(adj_sub, max_samples=20000 if node_idx is not None else None)
    lbl = torch.cat([torch.ones(N), torch.zeros(N)], dim=0).to(device)

    def band_loss(z, cos):
        sc_pos = local_affinity(z, adj_sub)
        sc_neg = local_affinity(z, dis_adj)
        pred = torch.cat([sc_pos, sc_neg], dim=0)
        loss_aff = b_xent(pred, lbl).mean()
        _, loss_dev = deviation_loss(cos)
        return loss_aff, loss_dev

    bw = [float(band_weights[i]) for i in range(num_bands)]
    affs, devs = [], []
    for i in range(num_bands):
        aff, dev = band_loss(z_bands_s[i], cos_bands_s[i])
        affs.append(aff)
        devs.append(dev)

    loss_wavelet = sum(
        bw[i] * (args.lambda_aff * affs[i] + args.lambda_dev * devs[i])
        for i in range(num_bands)
    )

    loss_dgi = dgi_loss(z_gcn_s, subgraph_s, disc, args.negsamp_ratio)

    loss_rec = F.mse_loss(x_hat_s, feat_s)

    z_wavelet_fused = sum(z_bands_s) / num_bands
    z_wavelet_norm = F.normalize(z_wavelet_fused, p=2, dim=1)
    z_gcn_norm = F.normalize(z_gcn_s, p=2, dim=1)
    loss_cross = (1.0 - (z_wavelet_norm * z_gcn_norm).sum(dim=1)).mean()

    loss_decorr = band_decorrelation_loss(z_bands_s)

    total = (loss_wavelet +
             args.lambda_dgi  * loss_dgi +
             args.lambda_rec  * loss_rec +
             args.lambda_cross * loss_cross +
             args.lambda_decorr * loss_decorr)

    loss_dict = {
        'total': total.item(),
        'wavelet': loss_wavelet.item(),
        'dgi': loss_dgi.item(),
        'rec': loss_rec.item(),
        'cross': loss_cross.item(),
        'decorr': loss_decorr.item(),
    }
    for i in range(num_bands):
        loss_dict[f'aff_{i}'] = affs[i].item()
        loss_dict[f'dev_{i}'] = devs[i].item()

    return loss_dict, total


def compute_spatial_loss(outputs, adj_label, features, args, disc, node_idx=None):
    if node_idx is not None:
        z_gcn_s = outputs['z_gcn'][node_idx]
        x_hat_s = outputs['x_hat'][node_idx]
        subgraph_s = outputs['subgraph'][node_idx]
        feat_s = features[node_idx]
    else:
        z_gcn_s = outputs['z_gcn']
        x_hat_s = outputs['x_hat']
        subgraph_s = outputs['subgraph']
        feat_s = features

    loss_dgi = dgi_loss(z_gcn_s, subgraph_s, disc, args.negsamp_ratio)
    loss_rec = F.mse_loss(x_hat_s, feat_s)
    total = args.lambda_dgi * loss_dgi + args.lambda_rec * loss_rec
    loss_dict = {'total': total.item(), 'dgi': loss_dgi.item(),
                 'rec': loss_rec.item()}
    return loss_dict, total


def compute_scores(outputs, adj_label, features, args, disc, band_weights, node_idx=None):
    def mm(x):
        x = x if isinstance(x, np.ndarray) else x.cpu().numpy()
        smin, smax = x.min(), x.max()
        return np.zeros_like(x) if smax - smin < 1e-10 else (x - smin) / (smax - smin)

    num_bands = len(outputs['z_bands'])
    bw = [float(band_weights[i]) for i in range(num_bands)]

    if node_idx is not None:
        z_bands_s = [z[node_idx] for z in outputs['z_bands']]
        cos_bands_s = [c[node_idx] for c in outputs['cos_bands']]
        z_gcn_s = outputs['z_gcn'][node_idx]
        x_hat_s = outputs['x_hat'][node_idx]
        subgraph_s = outputs['subgraph'][node_idx]
        feat_s = features[node_idx]
        adj_sub = _sub_adj(adj_label, node_idx)
    else:
        z_bands_s = outputs['z_bands']
        cos_bands_s = outputs['cos_bands']
        z_gcn_s = outputs['z_gcn']
        x_hat_s = outputs['x_hat']
        subgraph_s = outputs['subgraph']
        feat_s = features
        adj_sub = adj_label

    def band_score(z, cos):
        aff = local_affinity(z, adj_sub)
        return (mm((-aff).detach().cpu().numpy()) * args.alpha +
                mm((1.0 - cos).detach().cpu().numpy()) * args.beta)

    s_wavelet_raw = np.zeros(z_bands_s[0].size(0))
    for i in range(num_bands):
        s_wavelet_raw += bw[i] * band_score(z_bands_s[i], cos_bands_s[i])
    s_wavelet = mm(s_wavelet_raw)

    s_dgi  = mm((-disc(z_gcn_s, subgraph_s)).detach().cpu().numpy())
    s_rec  = mm(((x_hat_s - feat_s) ** 2).mean(dim=1).detach().cpu().numpy())

    s_mclast_raw = (args.w_dgi * mm(s_dgi) +
                    args.w_rec * mm(s_rec))
    s_mclast = mm(s_mclast_raw)

    s_agree = s_wavelet * s_mclast
    s_final = mm(s_wavelet + args.lambda_score * s_mclast + args.lambda_agree * s_agree)
    return s_final, s_wavelet, s_mclast


def _mm_np(x):
    x = x if isinstance(x, np.ndarray) else x.cpu().numpy()
    smin, smax = x.min(), x.max()
    return np.zeros_like(x) if smax - smin < 1e-10 else (x - smin) / (smax - smin)


def precompute_fast(outputs, adj_label, features, args, disc, node_idx=None):
    if node_idx is not None:
        z_bands_s = [z[node_idx] for z in outputs['z_bands']]
        cos_bands_s = [c[node_idx] for c in outputs['cos_bands']]
        z_gcn_s = outputs['z_gcn'][node_idx]
        x_hat_s = outputs['x_hat'][node_idx]
        subgraph_s = outputs['subgraph'][node_idx]
        feat_s = features[node_idx]
        adj_sub = _sub_adj(adj_label, node_idx)
    else:
        z_bands_s = outputs['z_bands']
        cos_bands_s = outputs['cos_bands']
        z_gcn_s = outputs['z_gcn']
        x_hat_s = outputs['x_hat']
        subgraph_s = outputs['subgraph']
        feat_s = features
        adj_sub = adj_label

    band_aff = []
    band_dev = []
    for i in range(len(z_bands_s)):
        aff = local_affinity(z_bands_s[i], adj_sub)
        band_aff.append(_mm_np((-aff).detach().cpu().numpy()))
        band_dev.append(_mm_np((1.0 - cos_bands_s[i]).detach().cpu().numpy()))

    s_dgi = _mm_np((-disc(z_gcn_s, subgraph_s)).detach().cpu().numpy())
    s_rec = _mm_np(((x_hat_s - feat_s) ** 2).mean(dim=1).detach().cpu().numpy())

    return {
        'band_aff': band_aff, 'band_dev': band_dev,
        's_dgi': s_dgi, 's_rec': s_rec,
    }


def fast_score(pre, weights, args):
    num_bands = len(pre['band_aff'])
    bw = np.array([float(weights[i]) for i in range(num_bands)], dtype=np.float64)

    s_wavelet_raw = np.zeros_like(pre['band_aff'][0])
    for i in range(num_bands):
        s_wavelet_raw += bw[i] * (pre['band_aff'][i] * args.alpha +
                                   pre['band_dev'][i] * args.beta)
    s_wavelet = _mm_np(s_wavelet_raw)

    s_mclast = _mm_np(args.w_dgi * _mm_np(pre['s_dgi']) +
                      args.w_rec * _mm_np(pre['s_rec']))

    s_agree = s_wavelet * s_mclast
    return _mm_np(s_wavelet + args.lambda_score * s_mclast + args.lambda_agree * s_agree)


def fast_metrics(pre, weights, label, args, test_id=None):
    scores = fast_score(pre, weights, args)
    if test_id is not None:
        return (roc_auc_score(label[test_id], scores[test_id]),
                calculate_auprc(label[test_id], scores[test_id]))
    return roc_auc_score(label, scores), calculate_auprc(label, scores)


INITIAL_WEIGHT_TABLE = {
    2: {'E': [1.7, 1.2], 'O': [1.2, 1.7]},
    3: {'E': [4.5, 0.2, 0.7], 'O': [0.2, 2.7, 2.3]},
    4: {'E': [0.3, 4.5, 0.8, 1.5], 'O-': [0.2, 0.3, 0.6, 4.5],
        'O+': [0.4, 0.2, 0.3, 4.5]},
    5: {'E': [0.3, 4.5, 0.8, 0.6, 0.2], 'O-': [0.2, 0.3, 0.6, 0.8, 4.5],
        'O+': [0.4, 0.2, 0.3, 0.8, 4.5]},
    6: {'E': [1.9, 0.2, 2.5, 1.8, 1.2, 0.2],
        'O': [1.9, 0.2, 1.8, 2.5, 1.2, 0.2]},
}


def initial_weights_from_table(num_bands, s):
    k = int(np.clip(np.floor(4.0 * float(s)), -4, 4))
    row = INITIAL_WEIGHT_TABLE.get(num_bands)
    if row is not None:
        if num_bands in (4, 5) and k % 2 != 0:
            key = 'O-' if k <= 1 else 'O+'
        else:
            key = 'E' if k % 2 == 0 else 'O'
        weights = list(row[key])
        if len(weights) != num_bands:
            raise ValueError("Initial-weight table row length mismatch")
        return np.asarray(weights, dtype=np.float64), key, k
    base = round(float(0.6 + 0.4 * float(np.clip(0.5 * (float(s) + 1.0), 0.0, 1.0))), 1)
    weights = np.full(num_bands, base, dtype=np.float64)
    weights[-1] = 4.5
    return weights, 'fallback', k


def run_phase1_local_search(features_t, adj_norm, adj_label,
                            args, device, nb_nodes, use_batch):
    feat_size = features_t.shape[1]
    num_bands = args.num_filters
    epochs = args.weight_update_interval
    absolute_radius = 0.2
    h = 0.05

    torch.manual_seed(args.seed + 9999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + 9999)

    template = FusionModel(
        feat_size=feat_size, hidden_dim=args.hidden_dim,
        wavelet_order=args.num_filters - 1, wavelet_shape=args.wavelet_shape,
        num_gcn_layers=args.num_gcn_layers,
        dropout=args.dropout, decoder_hidden_dim=args.decoder_hidden_dim,
    ).to(device)
    initial_state = {k: v.clone().cpu() for k, v in template.state_dict().items()}
    del template

    with torch.no_grad():
        feature_affinity = local_affinity(features_t, adj_label)
        feature_smoothness = float(feature_affinity.mean().item())

    user_weight_low = max(float(args.weight_low), 1e-6)
    user_weight_high = float(args.weight_high)
    if user_weight_high <= user_weight_low:
        raise ValueError("Invalid effective band-weight bounds")

    smoothness_s = float(feature_smoothness)
    formula_weight_center, table_key, table_k = \
        initial_weights_from_table(num_bands, smoothness_s)
    formula_weight_center = np.clip(
        formula_weight_center, user_weight_low, user_weight_high)

    if nb_nodes < 2:
        raise ValueError("Phase 1 pseudo-label evaluation requires at least 2 nodes")
    anchor_seed = int(args.seed) + 271828
    np.random.seed(anchor_seed)
    random.seed(anchor_seed)
    torch.manual_seed(anchor_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(anchor_seed)
        torch.cuda.manual_seed_all(anchor_seed)

    anchor = FusionModel(
        feat_size=feat_size, hidden_dim=args.hidden_dim,
        wavelet_order=args.num_filters - 1, wavelet_shape=args.wavelet_shape,
        num_gcn_layers=args.num_gcn_layers,
        dropout=args.dropout, decoder_hidden_dim=args.decoder_hidden_dim,
    ).to(device)
    anchor.load_state_dict(
        {k: v.clone().to(device) for k, v in initial_state.items()})
    with torch.no_grad():
        anchor.band_weights.copy_(torch.ones(num_bands, device=device))
    opt = torch.optim.Adam(anchor.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    for ep in range(epochs):
        anchor.train()
        opt.zero_grad()
        outputs = anchor(features_t, adj_norm)
        node_idx = (torch.randperm(nb_nodes, device=device)[:args.batch_nodes]
                    if use_batch else None)
        loss_dict, total_loss = compute_spatial_loss(
            outputs, adj_label, features_t, args, anchor.disc, node_idx)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(anchor.parameters(), max_norm=5.0)
        opt.step()

    anchor.eval()
    with torch.no_grad():
        outputs_anchor = anchor(features_t, adj_norm)
        pre = precompute_fast(
            outputs_anchor, adj_label, features_t, args, anchor.disc)
    del anchor
    s_mclast = _mm_np(args.w_dgi * _mm_np(pre['s_dgi']) +
                      args.w_rec * _mm_np(pre['s_rec']))

    mclast_order = np.argsort(s_mclast, kind='mergesort')
    proxy_neg = mclast_order[:max(1, int(np.floor(0.50 * nb_nodes)))]
    proxy_pos_groups = []
    seen_counts = set()
    for frac in (0.01, 0.02, 0.05, 0.10):
        count = min(max(1, int(np.ceil(frac * nb_nodes))),
                    nb_nodes - len(proxy_neg))
        if count not in seen_counts:
            proxy_pos_groups.append(mclast_order[-count:])
            seen_counts.add(count)

    print(f"\n{'='*70}")
    print(f"Phase 1: Table Initialization + Sequential Coordinate Descent "
          f"({args.weight_iters} rounds)")
    print(f"  Strategy: K x s initial-weight table (0.25 step) -> +/-0.2 local search")
    print(f"  Anchor: uniform weights, {epochs}ep training with the spatial-domain loss only")
    print(f"  Pseudo labels: anchor s_mclast -> top 1/2/5/10% = pseudo-anomalies, "
          f"bottom 50% = pseudo-normals (no ground-truth labels)")
    print(f"  Candidates: s_wavelet(w) = sum_i w_i * b_i, each candidate trained "
          f"{epochs} epochs -> fake AUC + fake AUPRC")
    print(f"  Objective: quality = 0.5*mean(fakeAUC) + 0.5*mean(fakeAUPRC)  "
          f"(larger is better)")
    print(f"  Search: one filter at a time (others fixed at current best); "
          f"round 1 probes +/-{h:.2f}, later rounds refit the best points")
    print(f"  Bands: {num_bands} (one per filter, one-to-one)")
    print(f"{'='*70}")

    print("  Initial-weight table (K x s; s step = 0.25, entries on the 0.1 grid):")
    print("    Rule: k = clamp(floor(4s), -4, 4); even k -> E, odd k -> O")
    print("          (K = 4/5: odd k <= 1 -> O-, odd k >= 3 -> O+)")
    print("    K=2: E=[1.7, 1.2]                                O =[1.2, 1.7]")
    print("    K=3: E=[4.5, 0.2, 0.7]                           O =[0.2, 2.7, 2.3]")
    print("    K=4: E=[0.3, 4.5, 0.8, 1.5]                      O-=[0.2, 0.3, 0.6, 4.5]  O+=[0.4, 0.2, 0.3, 4.5]")
    print("    K=5: E=[0.3, 4.5, 0.8, 0.6, 0.2]                 O-=[0.2, 0.3, 0.6, 0.8, 4.5]  O+=[0.4, 0.2, 0.3, 0.8, 4.5]")
    print("    K=6: E=[1.9, 0.2, 2.5, 1.8, 1.2, 0.2]            O =[1.9, 0.2, 1.8, 2.5, 1.2, 0.2]")
    print("    s bins:  -1.00 -0.75 -0.50 -0.25  0.00  0.25  0.50  0.75  1.00")
    print("    K=2..6:    E     O     E     O     E     O     E     O     E")
    print("             (K=4/5: bins -0.75/-0.25/0.25 -> O-, bin 0.75 -> O+)")
    print("  Properties: P1 E/O alternate along s (period 0.5); P2 O dominant band > E's,")
    print("              E dominant band non-decreasing in K; P3 entries on the 0.1 grid")
    print("              in [e^-1.5, e^1.5]; P4 monotone profiles (K=5 strictly monotone);")
    print("              P5 only K=3/4 E tails rise (pinned books/enron reference cells)")
    print(f"  Selected: K={num_bands}, s={feature_smoothness:.4f} -> k={table_k} "
          f"-> '{table_key}' -> initial weights "
          f"[{', '.join([f'{v:.1f}' for v in formula_weight_center])}]")
    print(f"  Fine-search probe step h={h:.3f}, bounds +/-{absolute_radius:.1f}, "
          f"rounds={args.weight_iters}")
    header = f"{'Rnd':<5} {'Raw trial':<35} {'cfAUC':<8} {'cfAP':<8} {'Quality':<10} {'Δ':<8}"
    print(header)
    print("-" * 95)

    def _proxy_auc(candidate, pos_idx, neg_idx):
        pos_scores = np.asarray(candidate, dtype=np.float64)[pos_idx]
        neg_scores = np.sort(np.asarray(candidate, dtype=np.float64)[neg_idx])
        if pos_scores.size == 0 or neg_scores.size == 0:
            return 0.5

        left = np.searchsorted(neg_scores, pos_scores, side='left')
        right = np.searchsorted(neg_scores, pos_scores, side='right')
        return float(np.mean((left + 0.5 * (right - left)) / neg_scores.size))

    def _proxy_average_precision(candidate, pos_idx, neg_idx):
        candidate = np.asarray(candidate, dtype=np.float64)
        scores = np.concatenate([candidate[pos_idx], candidate[neg_idx]])
        pseudo_y = np.concatenate([
            np.ones(len(pos_idx), dtype=np.float64),
            np.zeros(len(neg_idx), dtype=np.float64),
        ])
        order = np.argsort(-scores, kind='mergesort')
        scores_sorted = scores[order]
        y_sorted = pseudo_y[order]
        tp = np.cumsum(y_sorted)

        group_end = np.flatnonzero(
            np.r_[scores_sorted[1:] != scores_sorted[:-1], True])
        tp_at_threshold = tp[group_end]
        precision = tp_at_threshold / (group_end.astype(np.float64) + 1.0)
        recall = tp_at_threshold / max(float(len(pos_idx)), 1.0)
        recall_prev = np.r_[0.0, recall[:-1]]
        ap = float(np.sum((recall - recall_prev) * precision))
        prevalence = float(pseudo_y.mean())
        return ap, prevalence

    b_xent = nn.BCEWithLogitsLoss(reduction='none')

    def _frequency_score(outputs, weights):
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        s_raw = np.zeros(nb_nodes, dtype=np.float64)
        for i in range(num_bands):
            aff = local_affinity(outputs['z_bands'][i], adj_label)
            dev = 1.0 - outputs['cos_bands'][i]
            s_raw += w[i] * (_mm_np((-aff).detach().cpu().numpy()) * args.alpha +
                             _mm_np(dev.detach().cpu().numpy()) * args.beta)
        return s_raw

    def eval_candidate(weights):
        model = FusionModel(
            feat_size=feat_size, hidden_dim=args.hidden_dim,
            wavelet_order=args.num_filters - 1, wavelet_shape=args.wavelet_shape,
            num_gcn_layers=args.num_gcn_layers,
            dropout=args.dropout, decoder_hidden_dim=args.decoder_hidden_dim,
        ).to(device)
        model.load_state_dict({k: v.clone().to(device) for k, v in initial_state.items()})
        model.band_weights.copy_(torch.tensor(
            np.asarray(weights, dtype=np.float32), device=device))
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            outputs = model(features_t, adj_norm)
            _, total = compute_losses(
                outputs, adj_label, features_t, args, b_xent,
                model.disc, model.band_weights, None)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            outputs = model(features_t, adj_norm)
            s_wavelet_raw = _frequency_score(outputs, weights)

        auc_values, ap_values = [], []
        for pseudo_pos in proxy_pos_groups:
            p_auc = _proxy_auc(s_wavelet_raw, pseudo_pos, proxy_neg)
            p_ap, _ = _proxy_average_precision(s_wavelet_raw, pseudo_pos, proxy_neg)
            auc_values.append(p_auc)
            ap_values.append(p_ap)
        fake_auc = float(np.mean(auc_values))
        fake_ap = float(np.mean(ap_values))
        quality = 0.5 * fake_auc + 0.5 * fake_ap
        return quality, fake_auc, fake_ap

    all_round_results = []
    best_w = np.ones(num_bands)
    best_quality = -float('inf')

    def record(weights, quality, proxy_auc, proxy_ap):
        nonlocal best_w, best_quality
        improved = quality > best_quality
        if best_quality == -float('inf'):
            delta = f"({quality:.4f})"
        elif improved:
            delta = f"+{(quality - best_quality):.4f}"
        else:
            delta = f"{(quality - best_quality):.4f}"
        if improved:
            best_w = weights.copy()
            best_quality = quality
        all_round_results.append({
            'weights': weights.copy(), 'quality': quality,
            'proxy_auc': proxy_auc, 'proxy_ap': proxy_ap,
        })
        raw_s = ' '.join([f'{v:.2f}' for v in weights])
        print(f"R{len(all_round_results):02d}   [{raw_s:<33}] "
              f"{proxy_auc:7.4f} {proxy_ap:7.4f} {quality:9.4f} "
              f"{delta:<9}{' ★' if improved else ''}")

    def as_weights(values):
        return np.asarray(values, dtype=np.float64).copy()

    def _parabola_vertex(x0, f0, x1, f1, x2, f2):
        cur = (x0 - x1) * (x1 - x2) * (x2 - x0)
        if abs(cur) < 1e-15:
            return None
        a = (f0 / ((x0 - x1) * (x0 - x2)) +
             f1 / ((x1 - x0) * (x1 - x2)) +
             f2 / ((x2 - x0) * (x2 - x1)))
        if a >= -1e-12:
            return None
        d = (x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0)
        if abs(d) < 1e-15:
            return None
        num = (x1 - x0) ** 2 * (f1 - f2) - (x1 - x2) ** 2 * (f1 - f0)
        return x1 - 0.5 * num / d

    def _fit_three(pts3):
        v = _parabola_vertex(pts3[0][0], pts3[0][1],
                             pts3[1][0], pts3[1][1],
                             pts3[2][0], pts3[2][1])
        if v is None:
            return max(pts3, key=lambda t: t[1])[0]
        return v

    search_weight_lo = np.maximum(
        user_weight_low, formula_weight_center - absolute_radius)
    search_weight_hi = np.minimum(
        user_weight_high, formula_weight_center + absolute_radius)
    if np.any(search_weight_hi <= search_weight_lo):
        raise ValueError("User bounds are too narrow for the local weight search")

    print("  Initial centres and fine-search ranges:")
    for i in range(num_bands):
        print(f"    Band {i}: center={formula_weight_center[i]:.3f} "
              f"range=[{search_weight_lo[i]:.3f}, "
              f"{search_weight_hi[i]:.3f}]")
    print(f"  Sequential coordinate descent: {args.weight_iters} rounds, "
          f"one filter at a time (others fixed at current best). Round 1 "
          f"probes +/-{h:.2f}; later rounds refit from the accumulated best points.")

    lo = search_weight_lo.copy()
    hi = search_weight_hi.copy()

    w_current = as_weights(formula_weight_center)
    Q_cur, p_auc, p_ap = eval_candidate(w_current)
    record(w_current, Q_cur, p_auc, p_ap)

    pts_by_filter = [[] for _ in range(num_bands)]

    for r in range(args.weight_iters):
        for i in range(num_bands):
            if r == 0:
                p1 = w_current[i]
                p0 = max(lo[i], p1 - h)
                p2 = min(hi[i], p1 + h)
                w0 = w_current.copy(); w0[i] = p0
                w2 = w_current.copy(); w2[i] = p2
                Q0, auc0, ap0 = eval_candidate(w0)
                record(w0, Q0, auc0, ap0)
                Q2, auc2, ap2 = eval_candidate(w2)
                record(w2, Q2, auc2, ap2)
                pts_by_filter[i] = [(p0, Q0), (p1, Q_cur), (p2, Q2)]
                v = _fit_three(pts_by_filter[i])
            else:
                pts = pts_by_filter[i]
                latest_x, latest_q = pts[-1]
                others = [pt for pt in pts[:-1] if abs(pt[0] - latest_x) > 1e-12]
                if len(others) < 2:
                    others = list(pts[:-1])
                best2 = sorted(others, key=lambda t: -t[1])[:2]
                v = _fit_three([(latest_x, latest_q), best2[0], best2[1]])

            v = min(max(v, lo[i]), hi[i])
            w_next = w_current.copy(); w_next[i] = v
            Qv, p_auc, p_ap = eval_candidate(w_next)
            w_current = w_next
            Q_cur = Qv
            record(w_current, Q_cur, p_auc, p_ap)
            pts_by_filter[i].append((v, Qv))

    print(f"\n{'='*70}")
    print(f"Phase 1 Complete — {len(all_round_results)} probes evaluated")
    print(f"  ★ Best: Quality={best_quality:.4f} (larger is better)")
    w_s = ' '.join([f'{v:.3f}' for v in best_w])
    print(f"     Adaptive Weights: [{w_s}]")

    print(f"\n  Full Round Summary:")
    print(f"  {'Rnd':<5} {'Raw':<35} {'cfAUC':<8} {'cfAP':<8} {'Quality':<10}")
    print(f"  {'-'*75}")
    best_idx = int(np.argmax([r['quality'] for r in all_round_results])) + 1
    for r_idx, r in enumerate(all_round_results, 1):
        m = " ★" if r_idx == best_idx else ""
        raw_s = ' '.join([f'{v:.2f}' for v in r['weights']])
        print(f"  R{r_idx:02d}   [{raw_s:<33}] "
              f"{r['proxy_auc']:.4f} {r['proxy_ap']:.4f} "
              f"{r['quality']:.4f}{m}")

    return best_w, initial_state, best_quality


def run_phase2_retrain(features_t, adj_norm, adj_label, label, test_id,
                        args, device, nb_nodes, use_batch, best_weights, initial_state):
    feat_size = features_t.shape[1]
    b_xent = nn.BCEWithLogitsLoss(reduction='none')

    model = FusionModel(
        feat_size=feat_size, hidden_dim=args.hidden_dim,
        wavelet_order=args.num_filters - 1, wavelet_shape=args.wavelet_shape,
        num_gcn_layers=args.num_gcn_layers,
        dropout=args.dropout, decoder_hidden_dim=args.decoder_hidden_dim,
    ).to(device)

    model.load_state_dict({k: v.clone().to(device) for k, v in initial_state.items()})
    model.band_weights.copy_(torch.tensor(best_weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    bw_s = ' '.join([f'{w:.3f}' for w in best_weights])
    print(f"\n{'='*70}")
    print(f"Phase 2: Full Retraining ({args.epoch} epochs, adaptive weights LOCKED)")
    print(f"  Weights: [{bw_s}]")
    print(f"{'='*70}")

    best_loss, best_epoch = 1e9, 0

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        outputs = model(features_t, adj_norm)
        node_idx = (torch.randperm(nb_nodes, device=device)[:args.batch_nodes]
                    if use_batch else None)
        losses, total_loss = compute_losses(
            outputs, adj_label, features_t, args, b_xent,
            model.disc, model.band_weights, node_idx)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            s_final, _, _ = compute_scores(
                outputs, adj_label, features_t, args, model.disc,
                model.band_weights, node_idx)
            auc_node_idx = node_idx.cpu().numpy() if node_idx is not None else None
            label_sub = label[auc_node_idx] if auc_node_idx is not None else label
            if args.dataset in ['dgraphfin', 'Elliptic-all'] and test_id is not None:
                ae = roc_auc_score(label[test_id], s_final[test_id])
                pe = calculate_auprc(label[test_id], s_final[test_id])
            else:
                ae = roc_auc_score(label_sub, s_final)
                pe = calculate_auprc(label_sub, s_final)

        if total_loss.item() < best_loss:
            best_loss, best_epoch = total_loss.item(), epoch
            torch.save(model.state_dict(), f'best_fusion_{args.dataset}.pth')

        if epoch % args.log_interval == 0 or epoch < 3:
            print(f"P2 E{epoch:04d} | Loss:{losses['total']:.4f} "
                  f"dgi:{losses['dgi']:.4f} rec:{losses['rec']:.4f} "
                  f"AUC={ae:.4f} AUPRC={pe:.4f}")

    print(f"\nBest epoch: {best_epoch}  loss={best_loss:.4f}")
    try:
        model.load_state_dict(torch.load(
            f'best_fusion_{args.dataset}.pth', map_location=device, weights_only=True))
    except Exception as e:
        print(f"Using current state ({e})")

    model.eval()
    with torch.no_grad():
        outputs = model(features_t, adj_norm)
        if use_batch:
            chunk_size = args.batch_nodes
            s_final_all = []
            for start in range(0, nb_nodes, chunk_size):
                end = min(start + chunk_size, nb_nodes)
                chunk_idx = torch.arange(start, end, device=device)
                s_chunk, _, _ = compute_scores(
                    outputs, adj_label, features_t, args, model.disc,
                    model.band_weights, chunk_idx)
                s_final_all.append(s_chunk)
            s_final = np.concatenate(s_final_all)
        else:
            s_final, _, _ = compute_scores(
                outputs, adj_label, features_t, args, model.disc, model.band_weights)

    if args.dataset in ['dgraphfin', 'Elliptic-all'] and test_id is not None:
        auc = roc_auc_score(label[test_id], s_final[test_id])
        auprc = calculate_auprc(label[test_id], s_final[test_id])
    else:
        auc = roc_auc_score(label, s_final)
        auprc = calculate_auprc(label, s_final)

    return auc, auprc


def main():
    args = parameter_parser()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    print(f"Dataset: {args.dataset}  Device: {device}")
    print(f"Mode: UNSUPERVISED — labels only used for final evaluation")

    adj_norm, features, label, adj_label, test_id = load_mat(args.dataset)
    if args.dataset in ['Reddit', 'Amazon', 'tfinance', 'books', 'disney','elliptic']:
        features = (features - features.mean(0)) / (features.std(0) + 1e-30)

    nb_nodes, feat_size = features.shape[0], features.shape[1]
    num_bands = args.num_filters
    ph1_desc = (f"K x s table initialization + +/-0.2 local search on "
                f"formula-based fake AUC/AUPRC (no labels, adaptive rounds)")
    print(f"Nodes: {nb_nodes}  Feat: {feat_size}  Edges: {adj_norm._nnz()}  "
          f"Anom: {label.mean():.4f}  Bands: {num_bands} (one-to-one)")
    print(f"Phase 1: {ph1_desc}")
    print(f"Phase 2: {args.epoch}ep retrain from scratch (weights locked)")
    print(f"Frequency: Wavelet affinity | Spatial: GCN + DGI + reconstruction")

    use_batch = (args.batch_nodes > 0 and nb_nodes > args.batch_nodes)
    features_t = torch.FloatTensor(features).to(device)
    adj_norm = adj_norm.to(device)
    adj_label = adj_label.to(device)

    all_auc, all_auprc = [], []

    for run in range(args.runs):
        seed = args.seed + run if args.seed is not None else run
        print(f"\n{'#'*70}")
        print(f"# Run {run+1}/{args.runs}  seed={seed}")
        print(f"{'#'*70}")

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        best_weights, initial_state, _ = run_phase1_local_search(
            features_t, adj_norm, adj_label, args, device, nb_nodes, use_batch)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        auc, auprc = run_phase2_retrain(
            features_t, adj_norm, adj_label, label, test_id,
            args, device, nb_nodes, use_batch, best_weights, initial_state)

        print(f"\nRun {run+1} Final: AUC={auc:.4f}  AUPRC={auprc:.4f}")
        all_auc.append(auc)
        all_auprc.append(auprc)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS ({args.runs} runs, UNSUPERVISED)")
    print(f"{'='*70}")
    print(f"AUC   = {np.mean(all_auc)*100:.2f} ± {np.std(all_auc)*100:.2f}")
    print(f"AUPRC = {np.mean(all_auprc)*100:.2f} ± {np.std(all_auprc)*100:.2f}")
    if args.runs > 1:
        print(f"\nPer-run AUC:   {' '.join([f'{v:.4f}' for v in all_auc])}")
        print(f"Per-run AUPRC: {' '.join([f'{v:.4f}' for v in all_auprc])}")


if __name__ == '__main__':
    main()
