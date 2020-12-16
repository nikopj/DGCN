#!/usr/bin/env python3 
import os, sys, json
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import torch
from train import initModel
import utils, visual

def main(args):
	model_args, train_args, paths = [args[item] for item in ['model','train','paths']]
	ngpu = torch.cuda.device_count() 
	device = torch.device("cuda:0" if ngpu > 0 else "cpu")
	print(f"Using device {device}.")
	if ngpu > 1:
		print(f"Using {ngpu} GPUs.")
		data_parallel = True
	else:
		data_parallel = False
	model, _, _, epoch0 = initModel(args, device=device)
	model.eval()

	plotCurves(args, epoch0, show=False, save=True)

	#Dconv = model.D.Conv.weight.data.transpose(0,1)
	#visual.visplot(Dconv.cpu(), (8,8))

	x = utils.imgLoad("Set12/04.png", gray=True).to(device)
	y = utils.awgn(x, 25)
	with torch.no_grad():
		xhat, edge = model(y, ret_edge=True)
	print(edge.shape)
	psnr = (-10*torch.log10(torch.mean((x-xhat)**2))).item()
	print(f"PSNR = {psnr:.2f}")
	#fig1 = visual.visplot(torch.cat([y, xhat, x]).cpu())
	#fig2, handler = visual.visneighbors(xhat.cpu(), edge.cpu(), local_area=3)
	plt.show()

def plotCurves(args, epoch0, show=False, save=True):
	save_dir = args['paths']['save']
	curves = {}; exists = False
	# read log files
	for phase in ['train','val','test']:
		fn = os.path.join(save_dir, f"{phase}.psnr")
		if os.path.exists(fn):
			curves[phase] = np.loadtxt(fn, delimiter=',')
			exists = True
	if not exists: # exit if there are no log files
		return 1
	x = np.arange(1,epoch0+1)
	fig = plt.figure()
	if 'train' in curves:
		plt.plot(x, curves['train'][:epoch0], '-oc', mec='k')
	if 'val' in curves:
		val_freq = args['train']['fit']['val_freq']
		xval = x[::val_freq]; xval = xval[xval <= epoch0]
		plt.plot(xval, curves['val'], '-ob', mec='k')
	if 'test' in curves:
		plt.plot(epoch0, curves['test'], 'sr', mec='k', ms=10)
	plt.grid()
	plt.legend(list(curves.keys()))
	if show:
		plt.show()
	if save:
		fn = os.path.join(save_dir, f"train_plot.png")
		print(f"Saving training curves to {fn} ... ")
		fig.savefig(fn, dpi=300)
	plt.close(fig)
	return 0

if __name__ == "__main__":
	""" Load arguments dictionary from json file to pass to main.
	"""
	if len(sys.argv)<2:
		print('ERROR: usage: analyze.py [path/to/arg_file.json]')
		sys.exit(1)
	args_file = open(sys.argv[1])
	args = json.load(args_file)
	pprint(args)
	args_file.close()
	main(args)
