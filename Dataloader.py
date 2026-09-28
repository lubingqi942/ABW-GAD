import numpy as np
import scipy.sparse as sp
from utils import *
from scipy.io import loadmat
import torch
from utils import normalize, sparse_mx_to_torch_sparse_tensor, normalize_adj
from args import parameter_parser
import scipy.io as sio

args = parameter_parser()


def load_dgl_graph(dataset, datadir='.'):
    import dgl

    path = f'{datadir}/{dataset}' if '/' not in dataset else dataset
    graph_list, _ = dgl.load_graphs(path)
    g = graph_list[0]

    if 'feature' in g.ndata:
        feat = g.ndata['feature'].numpy()
    elif 'feat' in g.ndata:
        feat = g.ndata['feat'].numpy()
    elif 'features' in g.ndata:
        feat = g.ndata['features'].numpy()
    else:
        raise KeyError(f"Cannot find features in DGL graph. ndata keys: {list(g.ndata.keys())}")

    if 'label' in g.ndata:
        raw_label = g.ndata['label'].numpy()
    elif 'labels' in g.ndata:
        raw_label = g.ndata['labels'].numpy()
    else:
        raise KeyError(f"Cannot find labels in DGL graph. ndata keys: {list(g.ndata.keys())}")

    if raw_label.ndim == 2 and raw_label.shape[1] == 2:
        truth = raw_label[:, 1].flatten()
    else:
        truth = raw_label.flatten()

    src, dst = g.edges()
    num_nodes = g.num_nodes()
    adj = sp.csr_matrix(
        (np.ones(len(src), dtype=np.float32), (src.numpy(), dst.numpy())),
        shape=(num_nodes, num_nodes)
    )
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)

    adj_norm = normalize_adj(adj)
    adj_norm_t = sparse_mx_to_torch_sparse_tensor(adj_norm)
    adj_t = sparse_mx_to_torch_sparse_tensor(adj)

    feat = np.array(feat, dtype=np.float32)
    if sp.issparse(feat):
        feat = feat.toarray()

    return adj_norm_t, feat, truth, adj_t, None


def load_mat(dataset, datadir='dataset'):
    if dataset in ['tfinance', 'tsocial']:
        return load_dgl_graph(dataset, datadir='dataset')


    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')

    if 'Network' in data_mat:
        adj = data_mat['Network']
    elif 'A' in data_mat:
        adj = data_mat['A']
    elif 'homo' in data_mat:
        adj = data_mat['homo']
    else:
        adj = None
        for k in data_mat.keys():
            if k.startswith('__'):
                continue
            v = data_mat[k]
            if hasattr(v, 'shape') and len(v.shape) == 2 and v.shape[0] == v.shape[1]:
                adj = v
                break
        if adj is None:
            raise KeyError(f"Cannot find adjacency matrix in {dataset}.mat. Keys: {list(data_mat.keys())}")

    if 'Attributes' in data_mat:
        feat = data_mat['Attributes']
    elif 'X' in data_mat:
        feat = data_mat['X']
    elif 'features' in data_mat:
        feat = data_mat['features']
    else:
        raise KeyError(f"Cannot find feature matrix in {dataset}.mat")

    if 'Label' in data_mat:
        truth = data_mat['Label']
    elif 'gnd' in data_mat:
        truth = data_mat['gnd']
    elif 'label' in data_mat:
        truth = data_mat['label']
    else:
        raise KeyError(f"Cannot find label vector in {dataset}.mat")

    if args.dataset == 'Elliptic-all':
        test_id = data_mat['k'].flatten()
    else:
        test_id = None

    truth = truth.flatten()

    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_norm = normalize_adj(adj)
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    feat = sp.lil_matrix(feat).toarray()
    return adj_norm, feat, truth, adj, test_id
