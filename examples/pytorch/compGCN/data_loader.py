import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl 
from dgl.data import FB15k237Dataset
from collections import defaultdict
import argparse
import time
import os

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run
	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------
		
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus


class TrainDataset(Dataset):
	"""
	Training Dataset class.
	Parameters
	----------
	triples:	The triples used for training the model
	params:		Parameters for the experiments
	
	Returns
	-------
	A training Dataset class instance used by DataLoader
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
		trp_label		= self.get_label(label)

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label, None, None

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label
	
	def get_neg_ent(self, triple, label):
		def get(triple, label):
			pos_obj		= label
			mask		= np.ones([self.p.num_ent], dtype=np.bool)
			mask[label]	= 0
			neg_ent		= np.int32(np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
			neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))

			return neg_ent

		neg_ent = get(triple, label)
		return neg_ent

	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)


class TestDataset(Dataset):
	"""
	Evaluation Dataset class.
	Parameters
	----------
	triples:	The triples used for evaluating the model
	params:		Parameters for the experiments
	
	Returns
	-------
	An evaluation Dataset class instance used by DataLoader for model evaluation
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.int32(ele['label'])
		label		= self.get_label(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label
	
	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)


class Data(object):
    def __init__(self, params):
        self.p = params
        self.load_data()
        
    def load_data(self):
        self.p.num_ent = 14541
        self.p.num_rel = 237
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
        self.data = defaultdict(list)
        sr2o = defaultdict(set)
        data = FB15k237Dataset()
        data_dict = {'train': data.train, 'test': data.test, 'valid': data.valid}
        
        for split, value in data_dict.items():
            for i in range(value.shape[0]):
                sub, rel, obj = value[i, 0], value[i, 1], value[i, 2]
                self.data[split].append((sub, rel, obj))
                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.p.num_rel)].add(sub) #inverse
        
        self.data=dict(self.data)
        self.sr2o={k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)
        
        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = defaultdict(list)

        for (sub, rel), obj in self.sr2o.items():
            self. triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size = batch_size,
                shuffle = shuffle,
                num_workers = max(0, self.p.num_workers),
                collate_fn = dataset_class.collate_fn
            )
        
        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }

        #print(self.data_iter)