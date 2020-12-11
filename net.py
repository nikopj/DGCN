#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import knn, utils

class DGCN(nn.Module):
	""" Deep Graph Convolutional Network
	Missing:
		- proper weight initialization
		- preprocessing network, HPF network, alpha/beta multipliers
	"""
	def __init__(self, nic=1, nf=16, iters=3, window_size=32, topK=8, **kwargs):
		super(DGCN, self).__init__()
		self.GClayer0 = GClayer(nic, nf, window_size=32, topK=8, **kwargs)
		self.GClayer = nn.ModuleList([GClayer(nf,nf,window_size=32,topK=8,**kwargs) for _ in range(iters-1)])
		self.GCout = GraphConv(nf,nic,**kwargs)
		self.local_mask = knn.localMask(window_size, window_size, 3)
		self.window_size = window_size
		self.topK = topK
		self.iters = iters

	def pre_process(self, x):
		params = []
		# mean-subtract
		xmean = x.mean(dim=(2,3), keepdim=True)
		x = x - xmean
		params.append(xmean)
		# pad signal for windowed processing (in GraphConv)
		pad = utils.calcPad2D(*x.shape[2:], self.window_size)
		x = F.pad(x, pad, mode='reflect')
		params.append(pad)
		return x, params

	def post_process(self, x, params):
		# unpad
		pad = params.pop()
		x = utils.unpad(x, pad)
		# add mean
		xmean = params.pop()
		x = x + xmean
		return x

	def forward(self, x):
		x, params = self.pre_process(x)
		x = self.GClayer0(x)
		for i in range(self.iters-1):
			x = self.GClayer[i](x)
		edge = knn.windowedTopK(x, self.topK, self.window_size, self.local_mask)
		x = self.GCout(x, edge)
		x = self.post_process(x, params)
		return x
		
class GClayer(nn.Module):
	def __init__(self, nic, noc, window_size=32, topK=8, **kwargs):
		super(GClayer, self).__init__()
		self.window_size = window_size
		self.topK = topK
		self.local_mask = knn.localMask(window_size, window_size, 3)
		self.Conv = nn.Conv2d(nic, noc, 3, padding=1, padding_mode='reflect', bias=True)
		self.BN0 = nn.BatchNorm2d(noc)
		self.BN = nn.ModuleList([nn.BatchNorm2d(noc) for _ in range(3)])
		self.GConv = nn.ModuleList([GraphConv(noc, noc, **kwargs) for _ in range(3)])

	def forward(self, x):
		x = F.relu(self.BN0(self.Conv(x)))
		edge = knn.windowedTopK(x, self.topK, self.window_size, self.local_mask)
		for i in range(3):
			x = self.GConv[i](x, edge)
			x = self.BN[i](x)
			x = F.relu(x)
		return x

class GraphConv(nn.Module):
	""" Graph-Convolution Module.
	Computes average (+bias) of learned local and non-local convolutions.
	"""
	def __init__(self, Cin, Cout, ks=3, **kwargs):
		super(GraphConv, self).__init__()
		self.NLAgg = LowRankECC(Cin, Cout, **kwargs)
		self.Conv  = nn.Conv2d(Cin, Cout, ks, padding=(ks-1)//2, padding_mode='reflect', bias=False)
		self.bias  = nn.Parameter(torch.zeros(1,Cout,1,1))

	def forward(self, h, edge):
		""" 
		h: (B, C, H, W) batched input feature map for image shape (H, W)
		edge: (B, K, H, W) edge indices for K-Regular-Graph of nearest neighbors for each pixel
		"""
		hNL  = self.NLAgg(h, edge)
		hL   = self.Conv(h)
		return (hNL + hL)/2 + self.bias

class LowRankECC(nn.Module):
	""" Low Rank Edge-Conditioned Convolution, 2-layer MLP
	"""
	def __init__(self, Cin, Cout, rank=10, delta=10, leak=1e-2):
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
	
	def forward(self, h, edge):
		"""
		h: (B, C, H, W) input image
		edge: (B, K, H, W) indices of K connected verticies 
		output: (B, Cout, H, W), edge-conditioned non-local conv
		"""
		B, K, H, W = edge.shape
		N = H*W
		# (B, K, N, C)
		# get labels and vertex-set associated with each pixel
		label, vertex = knn.getLabelVertex(h, edge)
		# move pixels and K neighbors into batch dimension
		label_tilde  = label.reshape(-1, self.Cin)
		vertex_tilde = vertex.reshape(-1, self.Cin)
		# layer-1: learned edge-label preprocess
		theta  = self.act(self.FC0(label_tilde))
		# layer-2: generate low-rank matrix for each neighbor based on edge-label
		B0 = B*K*N
		thetaL = self.FCL(theta).reshape(B0, self.Cout, self.rank)
		thetaR = self.FCR(theta).reshape(B0, self.Cin,  self.rank)
		kappa  = self.FCk(theta) 
		# stage-3: apply low-rank matrix
		# (Cout, 1) = (Cout, rank) @ diag(rank, 1) @ (rank, Cin), batch supressed
		output = thetaL @ (kappa.unsqueeze(-1) * (thetaR.transpose(1,2) @ vertex_tilde.unsqueeze(-1)))
		# stage-4: non-local attention term (B0,1,1)
		gamma = torch.exp(-torch.sum(label_tilde**2, dim=1, keepdim=True)/self.delta).unsqueeze(-1)
		# average over K neighbors
		output = (gamma*output).reshape(B, K, N, self.Cout).mean(dim=1) # (B,N,Cout)
		# reshape to image 
		output = output.permute(0,2,1).reshape(B, self.Cout, H, W)
		return output
