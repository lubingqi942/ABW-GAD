import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='Reddit',
                        help='Dataset name: Reddit/Amazon/Facebook/YelpChi/elliptic/BlogCatalog/tfinance')
    parser.add_argument('--epoch', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 weight decay')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Print metrics every N epochs')

    parser.add_argument('--num_gcn_layers', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--decoder_hidden_dim', type=int, default=128,
                        help='Hidden dimension of reconstruction decoder MLP')
    parser.add_argument('--batch_nodes', type=int, default=0,
                        help='Nodes per batch for large graphs (0=full graph). Use 3000-5000 for tsocial.')

    parser.add_argument('--lambda_aff', type=float, default=1.0,
                        help='Weight for local affinity loss (MI-GAD core)')
    parser.add_argument('--lambda_dev', type=float, default=0.1,
                        help='Weight for centroid deviation loss (MI-GAD core)')
    parser.add_argument('--lambda_dgi', type=float, default=0.5,
                        help='Weight for DGI node-subgraph MI loss')
    parser.add_argument('--lambda_rec', type=float, default=0.25,
                        help='Weight for feature reconstruction MSE loss')
    parser.add_argument('--lambda_cross', type=float, default=0.5,
                        help='Weight for cross-modal consistency loss (GCN↔Wavelet alignment)')
    parser.add_argument('--lambda_decorr', type=float, default=0.5,
                        help='Weight for frequency-band decorrelation loss: pushes the '
                             'representations of different wavelet bands toward orthogonality '
                             '(squared cosine similarity), so bands stay complementary.')

    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for affinity score in wavelet band scoring')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Weight for deviation score in wavelet band scoring')
    parser.add_argument('--lambda_score', type=float, default=0.3,
                        help='Weight of complementary anomaly scores in final fusion')
    parser.add_argument('--lambda_agree', type=float, default=0.2,
                        help='Agreement bonus: boost nodes where wavelet and MCLAST both flag as anomalous')
    parser.add_argument('--w_dgi', type=float, default=1.0,
                        help='Weight for DGI mismatch anomaly score')
    parser.add_argument('--w_rec', type=float, default=0.5,
                        help='Weight for reconstruction error anomaly score')

    parser.add_argument('--negsamp_ratio', type=int, default=1,
                        help='Ratio of negative to positive samples for DGI discriminator')

    parser.add_argument('--num_filters', type=int, default=3,
                        help='Number of wavelet filters / bands. Each filter maps to one '
                             'independent frequency band with its own encoder '
                             '(wavelet_order = num_filters - 1 internally).')
    parser.add_argument('--wavelet_shape', type=float, default=1.0,
                        help='Beta wavelet shape (1.0=original BWGNN)')

    parser.add_argument('--weight_search', default='directional', choices=['directional', 'random'],
                        help='Weight-discovery algorithm. "directional" runs a coordinate-wise '
                             'quadratic-interpolation search in LOG space, anchored at all-ones '
                             '(w=1) and inferring each band\'s optimal weight from its probe '
                             'results; "random" keeps the original uniform random grid search '
                             'for ablation.')
    parser.add_argument('--weight_logstep', type=float, default=0.5,
                        help='[directional] Log-space probe step h: each band i is sampled at '
                             'u_i-h, u_i, u_i+h to fit a local parabola.')
    parser.add_argument('--weight_lr', type=float, default=0.5,
                        help='[directional] Trust-region / fallback step eta in log space: caps '
                             'the parabola-vertex update magnitude, and is the sign-step used '
                             'when the parabola has no usable downward curvature.')
    parser.add_argument('--weight_iters', type=int, default=3,
                        help='[Phase 1] Number of sequential coordinate-descent rounds: each '
                             'round updates every filter once (others fixed at their current '
                             'best), with a two-pass parabolic interpolation.')
    parser.add_argument('--weight_low', type=float, default=1e-3,
                        help='[directional] Lower bound of the band-weight search range.')
    parser.add_argument('--weight_high', type=float, default=1e3,
                        help='[directional] Upper bound of the band-weight search range.')

    parser.add_argument('--weight_update_rounds', type=int, default=20,
                        help='[random mode] Number of weight-search rounds in Phase 1 '
                             '(random sampling Uniform(0.1, 3.0) per band)')
    parser.add_argument('--weight_update_interval', type=int, default=5,
                        help='Epochs per weight-vector evaluation in Phase 1 '
                             '(both modes: 5 epochs -> 1 evaluation)')

    parser.add_argument('--device', default='cuda', type=str, help='Device: cuda / cpu')
    parser.add_argument('--gpu', type=int, default=2, help='GPU device ID (0, 1, 2, 3...)')

    args = parser.parse_args()

    if args.num_filters < 2:
        raise ValueError(f"--num_filters must be >= 2, got {args.num_filters}")

    return args
