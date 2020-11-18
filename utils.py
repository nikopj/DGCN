import torch
import torch.nn.functional as F
import numpy as np

def calcPad1D(L, M):
	""" Return symmetric pad sizes for length L 1D signal 
	to be divided into non-overlapping windows of size M.
	"""
	if L%M == 0:
		Lpad = [0,0]
	else:
		Lprime = np.ceil(L/M) * M
		Ldiff  = Lprime - L
		Lpad   = [int(np.floor(Ldiff/2)), int(np.ceil(Ldiff/2))]
	return Lpad

def calcPad2D(H, W, M):
	""" Return pad sizes for image (H,W) to be 
	divided into windows of size (MxM).
	(H,W): input height, width
	M: window size
	output: (padding_left, padding_right, padding_top, padding_bottom)
	"""
	return (*calcPad1D(W,M), *calcPad1D(H,M))

def pad(I, pad):
	""" Reflection padding.
	"""
	return F.pad(I, pad, mode='reflect')

def unpad(I, pad):
	""" Remove padding from 2D signal I.
	"""
	if pad[3] == 0 and pad[1] > 0:
		return I[..., pad[2]:, pad[0]:-pad[1]]
	elif pad[3] > 0 and pad[1] == 0:
		return I[..., pad[2]:-pad[3], pad[0]:]
	elif pad[3] == 0 and pad[1] == 0:
		return I[..., pad[2]:, pad[0]:]
	else:
		return I[..., pad[2]:-pad[3], pad[0]:-pad[1]]

def stack(I, M):
	""" Stack I (B, C, H, W) into patches of size (MxM) in the 
	batch dimension. H, W are assumed to be divisible by M.
	S: (B*(H/M)*(W/M), C, M, M) patch-stacked output
	"""
	S = I.unfold(2,M,M).unfold(3,M,M)
	S = S.permute(0,2,3,1,4,5).reshape(-1,I.shape[1],M,M)
	return S

def unstack(S, M, outshape):
	""" Tile stacked 2D signal S (B*pr*pc, C, M, M) into original 
	signal of size outshape (B, C, H, W).
	"""
	B, C, H, W = outshape
	nr, nc = H//M, W//M
	I = S.reshape(B, nr*nc, C*M*M).permute(0,2,1)
	I = F.fold(I, (H,W), M, stride=M)
	return I
	
