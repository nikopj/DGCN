#!/usr/bin/env python3
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import net, utils, visual

def main():
	#x1 = utils.imgLoad('Set12/04.png')
	#x2 = utils.imgLoad('Set12/05.png')
	#x  = torch.cat([x1,x2])
	#x = utils.imgLoad('CBSD68/0018.png')
	x = utils.imgLoad('Set12/04.png')

	C = x.shape[1]
	Cout = 3
	rank = 3
	K = 8
	ks = 3
	M = 32

	pad = utils.calcPad2D(*x.shape[2:], M)
	xpad = F.pad(x, pad, mode='reflect')  # (B, C, H, W)

	B, C, H, W = xpad.shape
	N = H*W

	mask = localMask(M, M, ks)
	#edge = windowedTopK(xpad, K, M, mask)
	print("Finding local topK edges...")
	edge = localTopK(xpad, K, M, mask)
	print("done.")
	print(f"edge.shape = ")
	print(edge.shape)
	print(edge.dtype)

	# (B, K, N, C)
	label, vertex = getLabelVertex(xpad, edge)

	GConv = net.GraphConv(C,Cout, ks=ks)
	ypad = GConv(xpad, edge)
	y = utils.unpad(ypad, pad)

	fig, handler = visual.visplotNeighbors(xpad, edge, local_area=True, depth=3)
	plt.show()

def windowedTopK(h, K, M, mask):
	""" Returns top K feature vector indices for 
	h: (B, C, H, W) input feature
	M: window side-length
	mask: (H*W, H*W) Graph mask.
	output: (B, K, H, W) K edge indices (of flattened image) for each pixel
	"""
	# stack image windows
	hs = utils.stack(h, M)          # (B,I,J,C,M,M)
	I, J = hs.shape[1], hs.shape[2]
	# move stack to match dimension to build batched Graph Adjacency matrices
	hbs = utils.batch_stack(hs)     # (B*I*J,C,M,M)
	G = graphAdj(hbs, mask)         # (B*I*J, M*M, M*M)
	# find topK in each window, unbatch the stack, translate window-index to tile index
	# (B*I*J,M*M,K) -> (B*I*J,K,M*M) -> (B*I*J, K, M, M)
	edge = torch.topk(G, K, largest=False).indices.permute(0,2,1).reshape(-1,K,M,M)
	edge = utils.unbatch_stack(edge, (I,J)) # (B,I,J,K,M,M)
	return utils.indexTranslate(edge) # (B,K,H,W)

def localTopK(h, K, M, mask):
	B, C, H, W = h.shape
	m1, m2 = np.floor((M-1)/2), np.ceil((M-1)/2)
	edge = torch.empty(B,K,H,W)
	for i in range(H): # (p,q) indexes the top-left corner of the window
		p = int(np.clip(i-m1,0,H-M))
		for j in range(W):
			q = int(np.clip(j-m1,0,W-M))
			loc_window = h[:,:,p:p+M,q:q+M]
			v = h[:,:,i,j][...,None,None]
			dist = torch.sum((loc_window - v)**2, dim=1).reshape(-1,M*M)
			mi, mj = i-p, j-q              # index in local window coords
			# window coords local area
			mi = np.clip(np.arange(mi-1,mi+2),0,M-1).astype(np.int64)
			mj = np.clip(np.arange(mj-1,mj+2),0,M-1).astype(np.int64)
			mask = mi.reshape(-1,1)*M + mj.reshape(1,-1)               # window flattened idx local area
			dist[:,mask] = torch.tensor(np.inf)
			loc_edge = torch.topk(dist, K, largest=False).indices
			m, n = loc_edge//M, loc_edge%M # flat window index to coord
			m, n = m+p, n+q                # window coord to image coord
			edge[:,:,i,j] = W*m + n        # image coord to flat image index
	return edge.long()

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
	""" Return edge labels and verticies for each pixel in input, derived from edges.
	Edges correspond to K-Regular Graph.
	input: (B, C, H, W)
	edge: input (B, K, H, W), edge indices
	label, vertex_set: output (B, K, N, C)
	"""
	B, K, H, W = edge.shape
	C, N = input.shape[1], H*W
	v = input.reshape(B, C, N)
	edge = edge.reshape(B, K, N)
	# differentiate indices in the batch dimension,
	edge = edge + torch.arange(0,B,device=input.device).reshape(-1,1,1)*N
	# put pixels in batch dimension
	v  = v.permute(0,2,1).reshape(-1, C)          # (BN, C)
	vS = torch.index_select(v, 0, edge.flatten()) # (BKN, C)
	# correspond pixels to nonlocal neighbors
	v  = v.reshape(B, N, C)
	vS = vS.reshape(B, K, N, C)
	label = vS - v.unsqueeze(1) # (B, K, N, C)
	return label, vS

def receptiveField(coord, edge, depth):
	""" Return receptive field / neighbors pixel at specified coordinate 
	after iterating with depth > 1.
	coord: 2-tuple or flattened index of spatial dimension
	edge: (1,K,H,W) KNN list for each pixel in (H,W)
	depth: scalar for depth of network considering these connections
	"""
	m, n = coord
	neighbors = edge[0,:,m,n]
	if depth < 1:
		return neighbors
	cols = edge.shape[-1]
	for i in range(len(neighbors)):
		new_coord = (neighbors[i]//cols, neighbors[i]%cols)
		neighbors = torch.cat([neighbors, receptiveField(new_coord, edge, depth-1)])
	return neighbors

if __name__ == "__main__":
	main()

