import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl 
from dgl.data import FB15k237Dataset
from collections import defaultdict
import argparse
import time
import os

class LoadData(object):
    def __init__(self):
        self.load_data()

    def load_data(self):
        self.num_ent = 14541
        self.num_rel = 237
        self.embed_dim = 200

        data = FB15k237Dataset()
        self.graph = data[0]

        train_mask = self.graph.edata['train_mask']
        train_idx = torch.nonzero(train_mask, as_tuple = False).squeeze()
        val_mask = self.graph.edata['val_mask']
        val_idx = torch.nonzero(val_mask, as_tuple = False).squeeze()
        test_mask = self.graph.edata['test_mask']
        test_idx = torch.nonzero(test_mask, as_tuple = False).squeeze()

        sub, obj = self.graph.edges()
        train_set = torch.stack((sub[train_idx], self.graph.edata['etype'][train_idx], obj[train_idx]), 1).numpy()
        val_set = torch.stack((sub[val_idx], self.graph.edata['etype'][val_idx], obj[val_idx]), 1).numpy()
        test_set = torch.stack((sub[test_idx], self.graph.edata['etype'][test_idx], obj[test_idx]), 1).numpy()
        
        print(train_set[1])
        print(train_set[272116])
        sr2o = defaultdict(set)
        for i in range(train_idx.shape[0]):
            sub, rel, obj = train_set[i, 0], train_set[i, 1], train_set[i, 2]
            sr2o[(sub, rel)].add(obj)
        self.sr2o = {k: list(v) for k,v in sr2o.items()}

        for i in range(val_idx.shape[0]):
            sub, rel, obj = val_set[i, 0], val_set[i, 1], val_set[i, 2]
            sr2o[(sub, rel)].add(obj)

        for i in range(test_idx.shape[0]):
            sub, rel, obj = test_set[i, 0], test_set[i, 1], test_set[i, 2]
            sr2o[(sub, rel)].add(obj)

        self.sr2o_all = {k: list(v) for k,v in sr2o.items()}

        self.triples = defaultdict(list)
        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        
        for i in range(val_idx.shape[0]):
            sub, rel, obj = val_set[i, 0], val_set[i, 1], val_set[i, 2]
            self.triples['valid_tail'].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
            #self.triples['valid_head'].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
          

        

LoadData()