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

def stack(T, M):
	""" Stack I (B, C, H, W) into patches of size (MxM).
	output: (B, C, I, J, H, W).
	"""
	return T.unfold(2,M,M).unfold(3,M,M)

def batch_stack(S):
	""" Reorder stack (B, C, I, J, M, M) so that 
	patches are stacked in the batch dimension,
	output: (B*I*J, C, H, W)
	"""
	C, M = S.shape[1], S.shape[-1]
	return S.permute(0,2,3,1,4,5).reshape(-1,C,M,M)

def unbatch_stack(S, stack_shape):
	""" Reorder batched stack into non-batcheys)
	(B*I*J, C, M, M) -> (B, C, I, J, M, M)
	"""
	B, C, I, J, M, _ = stack_shape
	return S.reshape(B, I, J, C, M, M).permute(0,3,1,2,4,5)

def unstack(S):
	""" Tile patches to form image
	(B, C, I, J, M, M) -> (B, C, I*M, J*M)
	"""
	B, C, I, J, M, _ = S.shape
	T = S.reshape(B, I*J, C*M*M).permute(0,2,1)
	return F.fold(T, (I*M, J*M), M, stride=M)

def indexTranslate(idx, grid_idx=None):
	""" Translate stacked grid index (B, I, J, M*M, K)
	to tiled-image index, (B, H, W, K)
	"""
	B, I, J, MM, K = idx.shape
	M = int(np.sqrt(MM))
	# each idx entries grid-index
	grid_idx = torch.arange(0,I*J).repeat_interleave(M*M).reshape(1,I,J,M*M,1)
	# grid index row and column (inter-window)
	gi, gj = grid_idx//M, grid_idx%M
	# window index row and column (intra-window)
	wi, wj = idx//M, idx%M
	# global index row and column
	m, n = wi+gi*M, wj+ gj*M
	# global flattened index
	p = J*M*m + n
	# stack to tile (unstack requires float)
	return unstack(p.reshape(B,I,J,M,M,K).permute(0,5,1,2,3,4).float()).long().permute(0,2,3,1)

