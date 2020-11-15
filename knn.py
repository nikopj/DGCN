#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt

def imgLoad(path, gray=False):
	if gray:
		return to_tensor(Image.open(path).convert('L'))[None,...]
	return to_tensor(Image.open(path))[None,...]

def graphAdj(h, mask):
	""" ||h_j - h_i||^2 L2 similarity matrix formation
	Using the following identity:
		||h_j - h_i||^2 = ||h_j||^2 - 2h_j^Th_i + ||h_i||^2
	h (B, C, H, W)
	L (B, N, N), N=H*W
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

class LowRankECC(nn.Module):
	""" Low Rank Edge-Conditioned Convolution, 2-layer MLP
	"""
	def __init__(self, Cin, Cout, rank, delta=10, leak=1e-2):
		super(LowRankECC, self).__init__()
		self.FC0 = nn.Linear(Cin, Cin)       # edge-label preprocesser
		self.FCL = nn.Linear(Cin, rank*Cout) # left vector generator
		self.FCR = nn.Linear(Cin, rank*Cin)  # right vector generator
		self.FCk = nn.Linear(Cin, rank)      # scale generator
		self.act = nn.LeakyReLU(leak)
		self.rank  = rank
		self.Cin   = Cin
		self.Cout  = Cout
		self.delta = delta
	
	def forward(self, vertex, label):
		"""
		vertex: (B, Cin, N, K), K-neighbor vertex signals at pixel n
		labels (B, Cin, N, K), edge-labels for each K-neighbors at pixel n
		output (B, Cout, N), edge-conditioned non-local conv
		"""
		B, C, N, K = vertex.shape
		# move pixel and K neighbors into batch dimension
		label_tilde  = label.reshape(B*K, C, N).reshape(B*K*N, C)
		vertex_tilde = vertex.reshape(B*K, C, N).reshape(B*K*N, C)
		B0 = B*K*N
		# layer-1: learned edge-label preprocess
		theta  = self.act(self.FC0(label_tilde))
		# layer-2: generate low-rank matrix for each neighbor based on edge-label
		thetaL = self.FCL(theta).reshape(B0, self.Cout, self.rank)
		thetaR = self.FCR(theta).reshape(B0, self.Cin,  self.rank)
		kappa  = self.FCk(theta) 
		# stage-3: apply low-rank matrix
		# (Cout, 1) = (Cout, rank) @ diag(rank, 1) @ (rank, Cin), batch supressed
		output = thetaL @ (kappa.unsqueeze(-1) * (thetaR.transpose(1,2) @ vertex_tilde.unsqueeze(-1)))
		# stage-4: non-local attention term
		gamma = torch.exp(-torch.sum(label_tilde**2, dim=1, keepdim=True)/self.delta).unsqueeze(-1)
		# average over K neighbors
		output = (gamma*output).reshape(B*K, self.Cout, N).reshape(B, self.Cout, N, K).mean(dim=3)
		return output

class GraphConv(nn.Module):
	""" Graph-Convolution Module.
	Computes average (+bias) of learned local and non-local convolutions.
	"""
	def __init__(self, Cin, Cout, ks=3, rank=3, delta=10, leak=1e-2):
		super(GraphConv, self).__init__()
		self.NLAgg = LowRankECC(Cin, Cout, rank, delta, leak)
		self.Conv  = nn.Conv2d(Cin, Cout, ks, padding=(ks-1)//2, padding_mode='reflect', bias=False)
		self.bias  = nn.Parameter(torch.zeros(1,Cout,1,1))

	def forward(self, h, edge):
		""" 
		h: (B, C, H, W) batched input feature map for image shape (H, W)
		edge: (B, N, K) edge indices for K-Regular-Graph of nearest neighbors at pixel n of h
		"""
		B, C, H, W = h.shape
		label, vertex = getLabelVertex(h, edge)
		hNL  = self.NLAgg(vertex, label).reshape(B, C, H, W)
		hL   = self.Conv(h)
		return (hNL + hL)/2 + self.bias

x = imgLoad('CBSD68/0018.png')
print(x.shape)
C = 3
Cout = 3
rank = 3
K = 8
n = 165
m = 42
h = x[:,:,n:n+m,n:n+m]

B, C, H, W = h.shape
N = H*W

mask = localMask(m,m,3)
G = graphAdj(h, mask)
edge = torch.topk(G, K, largest=False).indices  # (B, N, K)

GConv = GraphConv(C,Cout)
hnew = GConv(h, edge)

a = hnew.min()
b = hnew.max()
print(a,b)
hnew = (hnew - a)/(b-a)

#plt.figure()
#plt.imshow(mask)
#plt.figure()
#plt.imshow(x.permute(0,2,3,1).squeeze())
plt.figure()
plt.imshow(h.permute(0,2,3,1).squeeze())
plt.figure()
plt.imshow(hnew.detach().permute(0,2,3,1).squeeze())
plt.figure()
plt.imshow(torch.clamp(G,0,1).squeeze())
#
plt.show()
