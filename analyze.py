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
	model = initModel(model_args, train_args, paths, device=device)[0]

	x = utils.imgLoad("Set12/09.png", gray=True).to(device)[:,:,128:128+128,128:128+128]
	y = utils.awgn(x, 25)
	xhat, edge = model(y, ret_edge=True)
	print(edge.shape)
	psnr = (-10*torch.log10(torch.mean((x-xhat)**2))).item()
	print(f"PSNR = {psnr:.2f}")
	fig1 = visual.visplot(torch.cat([y, xhat, x]).cpu())
	fig2, handler = visual.visneighbors(xhat.cpu(), edge.cpu(), local_area=3)
	plt.show()

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
