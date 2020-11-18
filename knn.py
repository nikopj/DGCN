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
	#x = imgLoad('CBSD68/0018.png')
	#x = torch.arange(0, 10**2).reshape(1,1,10,10).float() 
	#x = x

	#print(x)

	print(x.shape)
	C = 1
	Cout = 3
	rank = 3
	K = 2
	n = 165
	M = 5
	h = x[:,:,n:n+2*M,n:n+2*M]

	pad = utils.calcPad2D(*x.shape[2:], M)
	xpad = utils.pad(x, pad)                # (B, C, H, W)

	B, C, H, W = xpad.shape
	N = H*W

	xs = utils.stack(xpad, M)               # (B, C, I, J, M, M)
	I, J = xs.shape[2], xs.shape[3]
	print(xs.shape)
	xbs = utils.batch_stack(xs)             # (B*I*J, C, M, M)

	ys = utils.unbatch_stack(xbs, xs.shape)
	ypad = utils.unstack(ys)
	y = utils.unpad(ypad, pad)
	visplot(y, (1,len(y)))
	plt.show()
	sys.exit()

	mask = localMask(M, M, 3)
	G = graphAdj(xbs, mask) # (B*I*J, M*M, M*M)
	edge = torch.topk(G, K, largest=False, sorted=True).indices.permute(0,2,1).reshape(-1,K,M,M) # (B*I*J,K,M,M) //(B*I*J, M*M, K)
	edge = utils.unbatch_stack(edge, (B,K,I,J,M,M)) # (B,K,I,J,M,M)
	edge_prime = utils.indexTranslate(edge) # (B,K,H,W)

	print("edge")
	#print(edge[0,:,0,0,0,0])
	print(edge)
	print("edge_prime")
	print(edge_prime)
	#print(edge_prime[0,:,0,0])

	print("edge k=0")
	print(edge[0,0])
	print("edge k=1")
	print(edge[0,1])

	

	#B = 1
	#H, W = 3, 6
	#K, M = 2, 3
	#I, J = H//M, W//M

	#nedge = torch.randint(0,M*M, (B,K,I,J,M,M))
	#nedge_prime = utils.indexTranslate(nedge)
	#print(nedge)
	#print(nedge_prime)
	#print(torch.arange(0,H*W).reshape(1,1,H,W))
	#print(torch.arange(0,M*M).reshape(1,1,M,M))
	#sys.exit()



	#GConv = net.GraphConv(C,Cout)
	#ypad = GConv(xpad, edge_prime.reshape(B, N, K))
	#y = utils.unpad(ypad, pad)

	#a = y.min()
	#b = y.max()
	#y = (y - a)/(b-a)

	im = plt.imshow((xpad[0]/xpad.max()).repeat(3,1,1).permute(1,2,0).squeeze())
	#im = plt.imshow(xpad[0].permute(1,2,0).squeeze())
	fig = plt.gcf()
	ax = plt.gca()

	handler = EventHandler(fig, ax, im, edge_prime)
	plt.show()

	#visplot(x, (1,len(x)))
	#visplot(y, (1,len(y)))
	#plt.show()

class EventHandler:
	def __init__(self, fig, ax, im, edge):
		fig.canvas.mpl_connect('button_press_event', self.onpress)
		self.fig = fig
		self.ax = ax
		self.im = im
		self.data = im.get_array()
		self.edge = edge
		self.cols = self.data.shape[1]

	def onpress(self, event):
		if event.inaxes!=self.ax:
			return
		self.im.set_array(self.data)
		n, m = (int(round(c)) for c in (event.xdata, event.ydata))
		print(m,n)
		e    = self.edge[0,:,m,n]
		em, en = e//self.cols, e%self.cols
		hi_data = self.data.copy()
		print("Edges:", em, en)
		for k in range(len(e)):
			hi_data[em[k],en[k],:] = np.array([255, 0, 0])
		self.im.set_array(hi_data)
		#value = self.im.get_array()[xi,yi]
		#color = self.im.cmap(self.im.norm(value))
		plt.draw()

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

