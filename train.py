#!/usr/bin/env python3
import os, sys, json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
from net import DGCN, GCDLNet
from data import getDataLoaders
from utils import awgn

def main(args):
	""" Given argument dictionary, load data, initialize model, and fit model.
	"""
	model_args, train_args, paths = [args[item] for item in ['model','train','paths']]
	loaders = getDataLoaders(**train_args['loaders'])
	ngpu = torch.cuda.device_count() 
	device = torch.device("cuda:0" if ngpu > 0 else "cpu")
	print(f"Using device {device}.")
	if ngpu > 1:
		print(f"Using {ngpu} GPUs.")
		data_parallel = True
	else:
		data_parallel = False
	model, opt, sched, epoch0 = initModel(args, device=device)
	fit(model, opt, loaders,
	    sched       = sched, 
	    save_dir    = paths['save'],
	    start_epoch = epoch0 + 1,
	    device      = device,
	    data_parallel = data_parallel,
	    **train_args['fit'],
	    epoch_fun   = lambda epoch_num: saveArgs(args, epoch_num))

def fit(model, opt, loaders,
        sched = None,
	    epochs = 1,
	    device = torch.device("cpu"),
	    save_dir = None,
	    start_epoch = 1,
	    clip_grad = 1,
	    noise_std = 25,
	    verbose = True,
	    val_freq  = 1,
	    save_freq = 1,
	    data_parallel = False,
	    epoch_fun = None,
	    backtrack_thresh = 1):
	""" fit model to training data.
	"""
	print(f"fit: using device {device}")
	print("Saving initialization to 0.ckpt")
	path = os.path.join(save_dir, '0.ckpt')
	saveCkpt(path, model, 0, opt, sched)
	top_psnr = {"train": 0, "val": 0, "test": 0} # for backtracking
	epoch = start_epoch
	while epoch < start_epoch + epochs:
		for phase in ['train', 'val', 'test']:
			model.train() if phase == 'train' else model.eval()
			if epoch != epochs and phase == 'test':
				continue
			if phase == 'val' and epoch%val_freq != 0:
				continue
			psnr = 0
			t = tqdm(iter(loaders[phase]), desc=phase.upper()+'-E'+str(epoch), dynamic_ncols=True)
			for itern, batch in enumerate(t):
				batch = batch.to(device)
				noisy_batch = awgn(batch, noise_std)
				opt.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					if phase == 'train' and data_parallel:
						output = nn.parallel.data_parallel(model, noisy_batch)
					else:
						output = model(noisy_batch)
					loss = torch.mean((batch-output)**2)
					if phase == 'train':
						loss.backward()
						if clip_grad is not None:
							nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
						opt.step()
				loss = loss.item()
				if verbose:
					total_norm = grad_norm(model.parameters())
					t.set_postfix_str(f"loss={loss:.1e}|gnorm={total_norm:.1e}")
				psnr = psnr - 10*np.log10(loss)
			psnr = psnr/(itern+1)
			print(f"{phase.upper()} PSNR: {psnr:.3f} dB")
			if psnr > top_psnr[phase]:
				top_psnr[phase] = psnr
			# backtracking check
			elif (psnr + backtrack_thresh < top_psnr[phase]) or np.isnan(loss) or np.isinf(loss):
				break
			with open(os.path.join(save_dir, f'{phase}.psnr'),'a') as psnr_file:
				psnr_file.write(f'{psnr}  ')
		if (psnr + backtrack_thresh < top_psnr[phase]) or np.isnan(loss) or np.isinf(loss):
			if epoch % save_freq == 0:
				epoch = epoch - save_freq
			else:
				epoch = epoch - epoch%save_freq
			old_lr = np.array(getlr(opt))
			print(f"Model has diverged. Backtracking to {epoch}.ckpt ...")
			path = os.path.join(save_dir, str(epoch) + '.ckpt')
			model, _, _, _ = loadCkpt(path, model, opt, sched)
			new_lr = old_lr * 0.8
			print("Updated Learning Rate(s):", new_lr)
			setlr(opt, new_lr)
			epoch = epoch + 1
			continue
		if sched is not None:
			sched.step()
			if hasattr(sched, "step_size") and epoch % sched.step_size == 0:
				print("Updated Learning Rate(s): ")
				print(getlr(opt))
		if epoch % save_freq == 0:
			path = os.path.join(save_dir, str(epoch) + '.ckpt')
			print('Checkpoint: ' + path)
			saveCkpt(path, model, epoch, opt, sched)
			if epoch_fun is not None:
				epoch_fun(epoch)
		epoch = epoch + 1

def grad_norm(params):
	""" computes norm of mini-batch gradient
	"""
	total_norm = 0
	for p in params:
		param_norm = torch.tensor(0)
		if p.grad is not None:
			param_norm = p.grad.data.norm(2)
		total_norm = total_norm + param_norm.item()**2
	return total_norm**(.5)
def getlr(opt):
	return [pg['lr'] for pg in opt.param_groups]
def setlr(opt, lr):
	if not issubclass(type(lr), (list, np.ndarray)):
		lr = [lr for _ in range(len(opt.param_groups))]
	for (i, pg) in enumerate(opt.param_groups):
		pg['lr'] = lr[i]

def initModel(args, device=torch.device("cpu")):
	""" Return model, optimizer, scheduler with optional initialization 
	from checkpoint.
	"""
	model_type, model_args, train_args, paths = [args[item] for item in ['type','model','train','paths']]
	if model_type == "DGCN":
		model = DGCN(**model_args)
	elif model_type == "GCDLNet":
		model = GCDLNet(**model_args)
	else:
		raise NotImplementedError
	model.to(device)
	initDir = lambda p: os.mkdir(p) if not os.path.isdir(p) else None
	initDir(os.path.dirname(paths['save']))
	initDir(paths['save'])
	opt = torch.optim.Adam(model.parameters(), **train_args['opt'])
	sched = torch.optim.lr_scheduler.StepLR(opt, **train_args['sched'])
	ckpt_path = paths['ckpt']
	if ckpt_path is not None:
		print(f"Initializing model from {ckpt_path} ...")
		model, opt, sched, epoch0 = loadCkpt(ckpt_path, model, opt, sched)
	else:
		epoch0 = 0
	print("Current Learning Rate(s):")
	for param_group in opt.param_groups:
		print(param_group['lr'])
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total Number of Parameters: {total_params:,}")
	return model, opt, sched, epoch0

def saveCkpt(path, model=None,epoch=None,opt=None,sched=None):
	""" Save Checkpoint.
	Saves model, optimizer, scheduler state dicts and epoch num to path.
	"""
	getSD = lambda obj: obj.state_dict() if obj is not None else None
	torch.save({'epoch': epoch,
	            'model_state_dict': getSD(model),
	            'opt_state_dict':   getSD(opt),
	            'sched_state_dict': getSD(sched)
	            }, path)

def loadCkpt(path, model=None,opt=None,sched=None):
	""" Load Checkpoint.
	Loads model, optimizer, scheduler and epoch number 
	from state dict stored in path.
	"""
	ckpt = torch.load(path, map_location=torch.device('cpu'))
	def setSD(obj, name):
		if obj is not None and name+"_state_dict" in ckpt:
			print(f"Loading {name} state-dict...")
			obj.load_state_dict(ckpt[name+"_state_dict"])
		return obj
	model = setSD(model, 'model')
	opt   = setSD(opt, 'opt')
	sched = setSD(sched, 'sched')
	return model, opt, sched, ckpt['epoch']

def saveArgs(args, epoch_num=None):
	""" Write argument dictionary to file, 
	with optionally writing the checkpoint.
	"""
	save_path = args['paths']['save']
	if epoch_num is not None:
		ckpt_path = os.path.join(save_path, f"{epoch_num}.ckpt")
		args['paths']['ckpt'] = ckpt_path
	with open(os.path.join(save_path, "args.json"), "+w") as outfile:
		outfile.write(json.dumps(args, indent=4, sort_keys=True))

if __name__ == "__main__":
	""" Load arguments dictionary from json file to pass to main.
	"""
	if len(sys.argv)<2:
		print('ERROR: usage: train.py [path/to/arg_file.json]')
		sys.exit(1)
	args_file = open(sys.argv[1])
	args = json.load(args_file)
	pprint(args)
	args_file.close()
	main(args)

