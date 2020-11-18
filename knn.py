#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import net
import utils
from visual import visplot
import sys

def main():
	x1 = imgLoad('Set12/04.png')
	x2 = imgLoad('Set12/05.png')
	x  = torch.cat([x1,x2])

	print(x.shape)
	C = 1
	Cout = 3
	rank = 3
	K = 8
	n = 165
	M = 32
	h = x[:,:,n:n+M,n:n+M]

	pad = utils.calcPad2D(*x.shape[2:], M)
	xpad = utils.pad(x, pad)                # (B, C, H, W)

	B, C, H, W = xpad.shape
	N = H*W

	xs = utils.stack(xpad, M)               # (B, C, I, J, M, M)
	I, J = xs.shape[2], xs.shape[3]
	print(xs.shape)
	xbs = utils.batch_stack(xs)             # (B*I*J, C, M, M)

	mask = localMask(M, M, 3)
	G = graphAdj(xbs, mask) # (B*I*J, M*M, M*M)
	edge = torch.topk(G, K, largest=False).indices # (B*I*J, M*M, K)
	edge = edge.reshape(B, I, J, M*M, K).float()
	edge_prime = utils.indexTranslate(edge) # (B, H, W, K)

	GConv = net.GraphConv(C,Cout)
	ypad = GConv(xpad, edge_prime.reshape(B, N, K))
	y = utils.unpad(ypad, pad)

	a = y.min()
	b = y.max()
	y = (y - a)/(b-a)

	visplot(x, (1,len(x)))
	visplot(y, (1,len(y)))
	#visplot(torch.clamp(G,0,1), (1, len(x)))
	plt.show()

def windowedTopK(h, M, K, mask):
	""" Returns top K feature vector indices for 
	h: (B, C, H, W) input feature
	M: window side-length
	mask: (H*W, H*W) Graph mask.
	output: (B, N, K) edge indices
	"""
	v = "yo"

def imgLoad(path, gray=False):
	if gray:
		return to_tensor(Image.open(path).convert('L'))[None,...]
	return to_tensor(Image.open(path))[None,...]

def graphAdj(h, mask):
	""" ||h_j - h_i||^2 L2 similarity matrix formation
	Using the following identity:
		||h_j - h_i||^2 = ||h_j||^2 - 2h_j^Th_i + ||h_i||^2
	h: input (B, C, H, W)
	mask: (H*W, H*W) 
	G: output (B, N, N), N=H*W
	"""
	N = h.shape[2]*h.shape[3] # num pixels
	v = h.reshape(-1,h.shape[1],N) # (B, C, N)
	vtv = torch.bmm(v.transpose(1,2), v) # batch matmul, (B, N, N)
	normsq_v = vtv.diagonal(dim1=1, dim2=2) # (B, N)
	# need to add normsq_v twice, with broadcasting in row/col dim
	G = normsq_v.unsqueeze(1) - 2*vtv + normsq_v.unsqueeze(2) # (B, N, N)
	# apply local mask (local distances set to infinity)
	G[:,~mask] = torch.tensor(np.inf)
	return G

def localMask(H,W,M):
	""" generate adjacency matrix mask to exclude local area around pixel
	H: image height
	W: image width
	M: local area side-length (square filter side-length)
	output: (H*W, H*W)
	"""
	N = H*W
	mask = torch.ones(N,N, dtype=torch.bool)
	m = (M-1)//2
	for pixel in range(N):
		for delta_row in range(-m,m+1):
			# absolute row number
			row = int(np.floor(pixel/W)) + delta_row
			# don't exit image
			if row < 0 or row > H-1:
				continue
			# clip local area to stop wrap-around
			a1 = int(np.clip(pixel%W - m, 0, W-1))
			a2 = int(np.clip(pixel%W + m, 0, W-1))
			local_row = np.arange(row*W + a1, row*W + a2 + 1) # local area of row
			mask[pixel, local_row] = False
	return mask

def getLabelVertex(input, edge):
	""" Return edge indices and labels of K-Regular-Graph derived from graph G
	input: (B, C, H, W)
	edge: output (B, N, K), edge indices
	label: output (B, C, N, K)
	vertex: output (B, C, N, K)
	"""
	B, N, K = edge.shape
	C = input.shape[1]
	v = input.reshape(B, C, N)
	edge_tilde = edge.reshape(edge.shape[0],1,-1).repeat(1,C,1) # (B, C, NK)
	# vS[i,j,k] = v[i, j, edge_tilde[i,j,k]]
	vS = torch.gather(v, 2, edge_tilde) # (B, C, NK)
	vertex = vS.reshape(*v.shape, K)
	label = vertex - v.unsqueeze(-1) # (B, C, N, K)
	return label, vertex

if __name__ == "__main__":
	main()

