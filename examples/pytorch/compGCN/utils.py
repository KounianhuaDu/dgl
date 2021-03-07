# This file is copied from the CompGCN author's implementation
# <https://github.com/malllabiisc/CompGCN/blob/master/helper.py>.
# It implements the operation of circular convolution in the ccorr function.

import torch


def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)


def conj(a):
	a[..., 1] = -a[..., 1]
	return a


def ccorr(a, b):
	"""
	Compute circular correlation of two tensors.
	Parameters
	----------
	a: Tensor, 1D or 2D
	b: Tensor, 1D or 2D

	Notes
	-----
	Input a and b should have the same dimensions. And this operation supports broadcasting.

	Returns
	-------
	Tensor, having the same dimension as the input a.
	"""
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def extract_cora_edge_direction(cora_graph):
	"""
	Extract the direction of edges in the cora graph data.

	:param cora_graph:
	:return:
	"""
	u, v = cora_graph.edges()

	edge_dict = {}
	for i, (src, dst) in enumerate(zip(u.tolist(), v.tolist())):
		if edge_dict.get((dst, src)):
			edge_dict[(dst, src)].append(i)
		else:
			edge_dict[(src, dst)] = [i]

	in_edges_mask = [False] * u.shape[0]
	out_edges_mask = [False] * u.shape[0]

	for k, v in edge_dict.items():
		in_edges_mask[v[0]] = True
		out_edges_mask[v[1]] = True

	return in_edges_mask, out_edges_mask
