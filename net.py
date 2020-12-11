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
		gckwargs = {"window_size": window_size, "topK": top_K}
		if nf % 3 != 0:
			raise ValueError(f"Expected nf {nf} to be divisible by 3.")
		self.INCONV = nn.ModuleList([nn.Conv2d(nic,nf//3,ks,padding=(ks-1)//2,padding_mode='reflect') for ks in [3,5,7]])
		self.PPCONV = nn.ModuleList([GClayer(nf//3,ks,block_type="PRE",**gckwargs,**kwargs) for ks in [3,5,7]])
		self.LPF = nn.ModuleList([GClayer(nf,3,block_type="LPF",**gckwargs,**kwargs) for _ in range(iters)])
		self.HPF = GClayer(nf,3,block_type="HPF",**gckwargs,**kwargs)
		### write ALPHAS AND BETAS
		self.GCout = GraphConv(nf,nic,3,**kwargs)
		self.wtkargs = (topK, window_size, knn.localMask(window_size, window_size, 3)) # windowedTopK args
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
		x = torch.cat([self.PPCONV[i](self.INCONV[i](x)) for i in range(3)], dim=1)
		for i in range(self.iters-1):
			x = self.GClayer[i](x)
		edge = knn.windowedTopK(x, *self.wtkargs)
		x = self.GCout(x, edge)
		x = self.post_process(x, params)
		return x
		
class GClayer(nn.Module):
	def __init__(self,nf,ks=3,window_size=32,topK=8,block_type="LPF",pre_iters=1,post_iters=3,**kwargs):
		super(GClayer, self).__init__()
		self.wtkargs = (topK, window_size, knn.localMask(window_size,window_size,ks)) # windowedTopK args
		self.Conv  = nn.ModuleList([nn.Conv2d(nf,nf,ks,padding=(ks-1)//2,padding_mode='reflect') for _ in range(pre_iters)])
		if block_type!="PRE":
			self.BNpre  = nn.ModuleList([nn.BatchNorm2d(nf) for _ in range(pre_iters)])
		if block_type=="LPF":
			self.BNpost = nn.ModuleList([nn.BatchNorm2d(nf) for _ in range(post_iters)])
		self.GConv = nn.ModuleList([GraphConv(nf,nf,ks,**kwargs) for _ in range(post_iters)])
		self.pre_iters  = pre_iters
		self.post_iters = post_iters
		self.block_type = block_type

	def forward(self, x):
		for i in range(self.pre_iters):
			x = self.Conv[i](x)
			if block_type!="PRE":
				x = self.BNpre[i](x)
			x = F.relu(x)
		edge = knn.windowedTopK(x, *self.wtkargs)
		for i in range(self.post_iters):
			x = self.GConv[i](x, edge)
			if block_type=="LPF":
				x = self.BNpost[i](x)
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
	def __init__(self, Cin, Cout, rank=12, delta=10, leak=1e-2, circ_rows=3):
		super(LowRankECC, self).__init__()
		self.FC0 = nn.Linear(Cin, Cin)       # edge-label preprocesser
		if circ_rows is None:
			Dense = nn.Linear; dkwargs = {}
		else:
			Dense = CircDense; dkwargs = {"m": circ_rows}
		self.FCL = Dense(Cin, rank*Cout, **dkwargs) # left vector generator
		self.FCR = Dense(Cin, rank*Cin,  **dkwargs) # right vector generator
		self.FCk = nn.Linear(Cin, rank)             # scale generator
		#self.FCk = Dense(Cin, rank, **dkwargs)      # scale generator
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

class CircDense(nn.Module):
	""" Circulant approximation of a dense matrix
	Treat this model as a nn.Linear module, except m determines the number 
	of circulant rows used (m=1 means a linear matrix). 
	Input (B, Cin)
	Output (B, Cout)
	"""
	def __init__(self, Cin, Cout, m=3):
		super(CircDense, self).__init__()
		if Cout % m != 0:
			raise ValueError(f"Expected Cout {Cout} to be divisible by m {m}.")
		self.weight = nn.Parameter(torch.randn(Cout//m,1,Cin))
		self.m = m
		self.Cout = Cout

	def forward(self, x):
		xpad = F.pad(x.unsqueeze(1), (0,self.m-1), mode='circular')
		return F.conv1d(xpad, self.weight).view(-1,self.Cout)





