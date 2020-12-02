import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

def imgLoad(path, gray=False):
	""" Load batched tensor image (1,C,H,W) from file path.
	"""
	if gray:
		return to_tensor(Image.open(path).convert('L'))[None,...]
	return to_tensor(Image.open(path))[None,...]

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
	output: (B, I, J, C, H, W).
	"""
	# (B,C,H,W) -> unfold (B,C,I,J,M,M) -> permute (B,I,J,C,M,M)
	return T.unfold(2,M,M).unfold(3,M,M).permute(0,2,3,1,4,5)

def batch_stack(S):
	""" Reorder stack (B, I, J, C, M, M) so that 
	patches are stacked in the batch dimension,
	output: (B*I*J, C, H, W)
	"""
	C, M = S.shape[3], S.shape[-1]
	return S.reshape(-1,C,M,M)

def unbatch_stack(S, grid_shape):
	""" Reorder batched stack into non-batcheys)
	(B*I*J, C, M, M) -> (B, I, J, C, M, M)
	"""
	I, J = grid_shape
	C, M = S.shape[1], S.shape[2]
	return S.reshape(-1, I, J, C, M, M)

def unstack(S):
	""" Tile patches to form image
	(B, I, J, C, M, M) -> (B, C, I*M, J*M)
	"""
	B, I, J, C, M, _ = S.shape
	T = S.reshape(B, I*J, C*M*M).permute(0,2,1)
	return F.fold(T, (I*M, J*M), M, stride=M)

def indexTranslate(idx):
	""" Translate stacked grid (flattened MxM window) index (B,I,J,K,M,M)
	to tiled-image (flattened HxW) index, (B,K,H,W)
	"""
	B, I, J, K, M, _ = idx.shape
	# each idx entries grid-index
	grid_idx = torch.arange(0,I*J).repeat_interleave(M*M).reshape(1,I,J,1,M,M).repeat_interleave(K, dim=3)
	# grid index row and column (inter-window)
	gi, gj = grid_idx//J, grid_idx%J
	# window index row and column (intra-window)
	wi, wj = idx//M, idx%M
	# global index row and column
	m, n = wi+gi*M, wj+gj*M
	# global flattened index
	p = J*M*m + n
	# stack to tile (unstack requires float)
	return unstack(p.float()).long()

