#!/usr/bin/env python3
import sys, json
from os.path import join
from pprint import pprint
import numpy as np

def write_args(arg_dict, name):
	with open(join("args",name+".json"), '+w') as outfile:
		outfile.write(json.dumps(arg_dict, indent=4, sort_keys=True))

args_file = open("args_gcdlnet.json")
args = json.load(args_file)
args_file.close()

loop_args = {
	"nf": [32, 64],
	"topK": [None, 8],
}

args["model"] = {
	"nic": 1,
	"nf": 64,
	"ks": 7,
	"iters": 10,
	"window_size": 32,
	"topK": 8,
	"rank": 11,
	"circ_rows": None,
	"leak": 0.2
}

args["train"] = {
	"loaders": {
		"batch_size": 4,
		"crop_size": 32,
		"load_color": False,
		"trn_path_list": ["CBSD432"],
		"val_path_list": ["Set12"],
		"tst_path_list": ["CBSD68"]
	},
	"fit": {
		"epochs": 3000,
		"noise_std": 25,
		"val_freq": 25,
		"save_freq": 5,
		"backtrack_thresh": 0.5,
		"verbose": False,
		"clip_grad": 5e-2
	},
	"opt": {
		"lr": 1e-3
	},
	"sched": {
		"gamma": 0.95,
		"step_size": 25
	}
}

args['type'] = "GCDLNet"
args['paths']['ckpt'] = None

vnum = 0
name = "nf_topK"

def product(*args, repeat=1):
	# product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
	# product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
	pools = [tuple(pool) for pool in args] * repeat
	result = [[]]
	for pool in pools:
		result = [x+[y] for x in result for y in pool]
	for prod in result:
		yield tuple(prod)

keys = list(loop_args.keys())

with open(f"Models/{args['type']}-{name}.summary", "a") as summary:
	for items in product(*[loop_args[k] for k in keys]):
		for i, it in enumerate(items):
			if keys[i] in args['model']:
				args['model'][keys[i]] = it
			elif keys[i] in args['train']['fit']:
				args['train']['fit'][keys[i]] = it

		version = args['type']+"-" + name + "-" + str(vnum)
		args['paths']['save'] = "Models/" + version
		write_args(args, version)
		print(f'{version}: {items}')
		summary.write(f'{version}: {items}\n')
		vnum += 1

