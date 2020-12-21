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

def awgn(input, noise_std):
	""" Additive White Gaussian Noise
	y: clean input image
	noise_std: (tuple) noise_std of batch size N is uniformly sampled 
	           between noise_std[0] and noise_std[1]. Expected to be in interval
			   [0,255]
	"""
	if not isinstance(noise_std, (list, tuple)):
		sigma = noise_std
	else: # uniform sampling of sigma
		sigma = noise_std[0] + \
		       (noise_std[1] - noise_std[0])*torch.rand(len(input),1,1,1, device=input.device)
	return input + torch.randn_like(input) * (sigma/255)

def pre_process(x, window_size, eval_phase):
	params = []
	# mean-subtract
	xmean = x.mean(dim=(2,3), keepdim=True)
	x = x - xmean
	params.append(xmean)
	# pad signal for windowed processing (in GraphConv)
	if eval_phase:
		pad = calcPad2D(*x.shape[2:], window_size)
		x = F.pad(x, pad, mode='reflect')
	else:
		pad = (0,0,0,0)
	params.append(pad)
	return x, params

def post_process(x, params):
	# unpad
	pad = params.pop()
	x = unpad(x, pad)
	# add mean
	xmean = params.pop()
	x = x + xmean
	return x

def calcPad1D(L, M):
	""" Return pad sizes for length L 1D signal to be divided by M
	"""
	if L%M == 0:
		Lpad = [0,0]
	else:
		Lprime = np.ceil(L/M) * M
		Ldiff  = Lprime - L
		Lpad   = [int(np.floor(Ldiff/2)), int(np.ceil(Ldiff/2))]
	return Lpad

def calcPad2D(H, W, M):
	""" Return pad sizes for image (H,W) to be divided by size M
	(H,W): input height, width
	output: (padding_left, padding_right, padding_top, padding_bottom)
	"""
	return (*calcPad1D(W,M), *calcPad1D(H,M))

def conv_pad(x, ks, mode):
	""" Pad a signal for same-sized convolution
	"""
	pad = (int(np.floor((ks-1)/2)), int(np.ceil((ks-1)/2)))
	return F.pad(x, (*pad, *pad), mode=mode)

def unpad(I, pad):
	""" Remove padding from 2D signalstack"""
	if pad[3] == 0 and pad[1] > 0:
		return I[..., pad[2]:, pad[0]:-pad[1]]
	elif pad[3] > 0 and pad[1] == 0:
		return I[..., pad[2]:-pad[3], pad[0]:]
	elif pad[3] == 0 and pad[1] == 0:
		return I[..., pad[2]:, pad[0]:]
	else:
		return I[..., pad[2]:-pad[3], pad[0]:-pad[1]]

def stack(T, ks, stride, padding=False):
	""" Stack I (B, C, H, W) into patches of size (MxM).
	output: (B, I, J, C, H, W).
	"""
	# (B,C,H,W) -> unfold (B,C,I,J,ks,ks) -> permute (B,I,J,C,ks,ks)
	return T.unfold(2,ks,stride).unfold(3,ks,stride).permute(0,2,3,1,4,5)

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

def indexTranslate(idx,M):
	""" Translate stacked grid (flattened MxM window) index (B,I,J,K,S,S)
	to tiled-image (flattened HxW) index, (B,K,H,W)
	"""
	B, I, J, K, S, _ = idx.shape
	# each idx entries grid-index
	grid_idx = torch.arange(0,I*J,device=idx.device).repeat_interleave(S*S).reshape(1,I,J,1,S,S).repeat_interleave(K, dim=3)
	# grid index row and column (inter-window)
	gi, gj = grid_idx//J, grid_idx%J
	# window index row and column (intra-window)
	#wi, wj = idx//S, idx%S
	wi, wj = idx//M, idx%M
	# global index row and column
	m, n = wi+gi*S, wj+gj*S
	# global flattened index
	p = J*S*m + n
	# stack to tile (unstack requires float)
	return unstack(p.float()).long()
	
