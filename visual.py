import torch
import numpy as np
from matplotlib import pyplot as plt
import knn

def visplot(images,
			grid_shape=None,
			crange = (None,None),
			primary_axis = 0,
			titles   = None,
			colorbar = False,
			ret_fig_im_ax = False):
	""" Visual Subplot, adapted from Amir's code.
	Plots array of images in grid with shared axes.
	Very nice for zooming.
	"""
	if grid_shape is None:
		grid_shape = (1,len(images))
	fig, axs = plt.subplots(*grid_shape,
							sharex='all',
							sharey='all',
							squeeze=False)
	nrows, ncols = grid_shape
	# fill grid row-wise or column-wise
	if primary_axis == 1:
		indfun = lambda i,j: j*nrows + i
	else:
		indfun = lambda i,j: i*ncols + j
	im_list = []
	for ii in range(nrows):
		for jj in range(ncols):
			ind = indfun(ii,jj)
			if ind < len(images):
				im = axs[ii,jj].imshow(images[ind].detach().permute(1,2,0).squeeze(),
									   cmap   = 'gray',
									   aspect = 'equal',
									   interpolation = None,
									   vmin = crange[0],
									   vmax = crange[1])
				im_list.append(im)
				if colorbar:
					fig.colorbar(im,
								 ax       = axs[ii,jj],
								 fraction = 0.046,
								 pad      = 0.04)
			axs[ii,jj].axis('off')
			if (titles is not None) and (ind < len(titles)):
				axs[ii,jj].set_title(titles[ind])
	if ret_fig_im_ax:
		return fig, im_list, axs
	return fig

class ClickHandler:
	def __init__(self, fig, ax, im, fun):
		fig.canvas.mpl_connect('button_press_event', self.onpress)
		self.fig = fig
		self.ax = ax
		self.im = im
		self.data = im.get_array()
		self.fun = fun

	def onpress(self, event):
		if event.inaxes!=self.ax:
			return
		self.im.set_array(self.data)
		i,j = (int(round(c)) for c in (event.ydata, event.xdata))
		new_data = self.fun((i,j), self.data.copy())
		self.im.set_array(new_data)
		plt.draw()

def visneighbors(edge, image=None, fig_im_ax=None, local_area=None, depth=0):
	""" Interactively visualize nonlocal neighbors of a 
	selected pixel on image (1,C,H,W) with nonlocal neighbors (1,K,H,W).
	Local neighborhood optionally displayed given side-length of local area (scalar).
	WARNING: Must set variables of return statement for ClickHandler to work!
	"""
	if image is not None:
		if image.shape[1]==1:
			image = image.repeat(1,3,1,1)
		fig = plt.figure()
		im  = plt.imshow(image[0].detach().permute(1,2,0).squeeze())
		ax  = plt.gca()
	elif fig_im_ax is not None:
		fig, im, ax = fig_im_ax
	def highlight(click_coord, data):
		cols = data.shape[1]
		i, j = click_coord
		neighbors = knn.receptiveField(click_coord, edge, depth)
		m, n = neighbors//cols, neighbors%cols
		alpha = 0.2
		data[m,n,:] = alpha*data[m,n,:] + (1-alpha)*np.array([0,1,0])
		if local_area is not None:
			rows = data.shape[0]
			p = (local_area-1)//2
			i1, i2 = np.clip(i-p,0,rows), np.clip(i+p+1,0,rows)
			j1, j2 = np.clip(j-p,0,cols), np.clip(j+p+1,0,cols)
			data[i1:i2,j1:j2,:] = np.array([1,1,0]) # local neighbors yellow
		data[i,j,:] = np.array([1,0,0]) # selected pixel red
		return data
	handler = ClickHandler(fig, ax, im, highlight)
	return fig, handler

def visplotNeighbors(image, edge, local_area=None, depth=0):
	""" Interactively visualize nonlocal neighbors of a 
	selected pixel on image (1,C,H,W) with nonlocal neighbors (1,K,H,W).
	Local neighborhood optionally displayed given side-length of local area (scalar).
	WARNING: Must set variables of return statement for ClickHandler to work!
	"""
	if image.shape[1]==1:
		image = image.repeat(1,3,1,1)
	if image.shape[0]==1:
		image = image.repeat(depth+1,1,1,1)
	fig, im_list, axs = visplot(image, ret_fig_im_ax=True)
	hand_list = []
	for i in range(depth+1):
		fig, hand = visneighbors(edge, fig_im_ax=(fig, im_list[i], axs[0,i]), depth=i)
		hand_list.append(hand)
	return fig, hand_list

