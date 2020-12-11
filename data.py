from os import path, listdir
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

class MyDataset(data.Dataset):
	def __init__(self, root_dirs, transform, load_color=False):
		self.image_paths = []
		for cur_path in root_dirs:
			self.image_paths += [path.join(cur_path, file) \
				for file in listdir(cur_path) \
				if file.endswith(('tif','tiff','png','jpg','jpeg','bmp'))]
		self.image_list = []
		print(f"Loading {root_dirs}:")
		for i in tqdm(range(len(self.image_paths))):
			if load_color:
				self.image_list.append(Image.open(self.image_paths[i]))
			else:
				self.image_list.append(Image.open(self.image_paths[i]).convert('L'))
		self.root_dirs = root_dirs
		self.transform = transform
	def __len__(self):
		return len(self.image_paths)
	def __getitem__(self, idx):
		return self.transform(self.image_list[idx])

def getDataLoaders(trn_path_list=['CBSD432'],
	               val_path_list=['CBSD432'],
	               tst_path_list=['CBSD68'],
	               crop_size = 128,
	               batch_size = [10,1,1],
	               load_color = False):
	train_xfm = transforms.Compose([transforms.RandomCrop(crop_size),
	                                transforms.RandomHorizontalFlip(),
	                                transforms.RandomVerticalFlip(),
	                                transforms.ToTensor()])
	test_xfm = transforms.ToTensor()
	if type(batch_size) is int:
		batch_size = [batch_size, 1, 1]
	dataloaders = {'train': data.DataLoader(MyDataset(trn_path_list, train_xfm, load_color),
	                                        batch_size = batch_size[0],
	                                        drop_last = True,
	                                        shuffle = True),
	               'test': data.DataLoader(MyDataset(tst_path_list, test_xfm, load_color),
	                                       batch_size = batch_size[1],
	                                       drop_last = False,
	                                       shuffle = False),
	               'val': data.DataLoader(MyDataset(val_path_list, test_xfm, load_color),
	                                      batch_size = batch_size[2],
	                                      drop_last = False,
	                                      shuffle = False)}
	return dataloaders
