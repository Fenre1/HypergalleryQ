import os
import glob
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from eval_cluster_utils import get_eval_args
from extractor import FeatureExtractionPipeline


def run_eval_pipeline(output_dir, out_dim, dataset_name, t):
    args = get_eval_args()
    args.dataset = dataset_name
    args.ckpt_folder = output_dir
    args.arch = ''
    args.num_workers = 0
    args.precomputed = True
    args.out_dim = out_dim 
    args.num_heads = 16
    args.embed_dim = 1536
    args.loader = 'EmbedNN'
    args.knn = 50
    args.loader_args = {}

    cudnn.deterministic = True

    checkpoint_list = glob.glob(os.path.join(args.ckpt_folder, "*.pth"))
    if not checkpoint_list:
        raise FileNotFoundError("No checkpoint found in the specified output_dir.")
    ckpt = checkpoint_list[0]
    # epoch = torch.load(ckpt, map_location='cpu')['epoch'] - 1
    # epoch = torch.load(ckpt, map_location='cpu', weights_only=False)['epoch'] - 1
    extractor = None
    if extractor is None or args.no_cache:
        extractor = FeatureExtractionPipeline(args, cache_backbone=not args.no_cache, datapath=args.datapath)
    train_features, test_features, train_labels, val_labels = extractor.get_features(ckpt)
    
    # Calculate max indices (for possible later use)
    _, max_indices = torch.max(test_features, dim=1)
    max_indices = max_indices.cpu().numpy()
    all_indices = test_features.cpu().numpy()

    def generate_hypergraph(all_indices, t):
        hypergraph = np.where(all_indices < t, 0, 1)
        for i in range(hypergraph.shape[0]):
            if np.all(hypergraph[i] == 0):
                hypergraph[i, np.argmax(all_indices[i])] = 1
        return hypergraph

    hypergraph = generate_hypergraph(all_indices, t)
    return hypergraph