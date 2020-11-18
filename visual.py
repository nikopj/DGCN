import torch
import numpy as np
from matplotlib import pyplot as plt

def visplot(images,
			grid_shape,
			crange = (None,None),
			primary_axis = 0,
			titles   = None,
			colorbar = False):
	""" Visual Subplot, adapted from Amir's code.
	Plots array of images in grid with shared axes.
	Very nice for zooming.
	"""
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
	for ii in range(nrows):
		for jj in range(ncols):
			ind = indfun(ii,jj)
			if ind < len(images):
				im = axs[ii,jj].imshow(images[ind].permute(1,2,0).squeeze(),
									   cmap   = 'gray',
									   aspect = 'equal',
									   interpolation = None,
									   vmin = crange[0],
									   vmax = crange[1])
				if colorbar:
					fig.colorbar(im,
								 ax       = axs[ii,jj],
								 fraction = 0.046,
								 pad      = 0.04)
			axs[ii,jj].axis('off')
			if (titles is not None) and (ind < len(titles)):
				axs[ii,jj].set_title(titles[ind])
	return fig
