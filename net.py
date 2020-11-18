#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import knn

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
		B, _, H, W = h.shape
		label, vertex = knn.getLabelVertex(h, edge)
		hNL  = self.NLAgg(vertex, label).reshape(B, -1, H, W)
		hL   = self.Conv(h)
		return (hNL + hL)/2 + self.bias
