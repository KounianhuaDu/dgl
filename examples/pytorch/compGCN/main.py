"""The main file to train a Simplified CompGCN model using a full graph."""

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from dgl.data import CoraGraphDataset
from dgl.data import FB15k237Dataset
import dgl.function as fn

from utils import ccorr, extract_cora_edge_direction

from models import CompGCN_TransE
from data_loader import Data



def main(args):

    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    '''
    if args.dataset == 'cora':
        dataset = CoraGraphDataset()
        graph = dataset[0]
    else:
        raise NotImplementedError
    '''

    # check cuda
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    
    data = Data(args)
    data_iter = data.data_iter
    graph = FB15k237Dataset()
    graph = graph[0]
    n_feats = nn.Parameter(torch.Tensor(graph.num_nodes(), 100))

    # Step 2: Create model =================================================================== #
    compgcn_model = CompGCN_TransE(100, 200, 200, 50, 474, 40.0)

    compgcn_model = compgcn_model.to(device)

    # Step 3: Create training components ===================================================== #
    #loss_fn = th.nn.CrossEntropyLoss()
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(compgcn_model.parameters(), lr=args.lr, weight_decay=5e-4)

    # Step 4: training epoches =============================================================== #
    for epoch in range(100):

        # Training and validation using a full graph
        compgcn_model.train()
        train_loss=[]
        for step, batch in enumerate(data_iter['train']):
            triple, label = [_.to(device) for _ in batch]
            sub, rel, obj, label = triple[:, 0], triple[:, 1], triple[:, 2], label
            logits = compgcn_model.forward(graph, n_feats, sub, rel)
		
            # compute loss
            tr_loss = loss_fn(logits[train_idx], labels[train_idx])
            train_loss.append(tr_loss)
            #tr_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)

            #valid_loss = loss_fn(logits[val_idx], labels[val_idx])
            #valid_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
        train_loss = np.sum(train_loss)
        # Print out performance
        print("In epoch {}, Train Loss: {:.4f}".format(epoch, train_loss))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
	parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')
	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

	parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
	parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
	parser.add_argument('-gpu',		type=int,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=500,  	help='Number of epochs')
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
	parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

	parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
	args = parser.parse_args()

	main(args)