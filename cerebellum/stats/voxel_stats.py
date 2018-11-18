import numpy as np
import h5py
import time
import matplotlib.pyplot as plt

"""
Functions to generate volume statistics from a segmentation
"""

def get_vols(label_data, ids=[], do_save=False, write_file="", do_hist=False):
	"""
	Returns voxel count of requested ids in segmentation
	Args:
		label_data (ndarray): input segmentation
		ids (list of ints)
		do_save (bool): flag to save result
		write_file (string): path to write result
		do_hist (bool): flag to generate histogram
	Returns:
		vols (ndarray): id #, voxel count sorted in descending order
	"""
	if len(ids)==0:
		unique_ids, counts = np.unique(label_data, return_counts=True)
	else:
		counts = [np.count_nonzero(label_data==id) for id in ids]
	if do_save:
		vols = np.array(zip(ids, counts), np.dtype(int,int))
		np.save(write_file,vols)
	if do_hist:
		plt.hist(x=vols[1:,1], bins=100) # remove segment 0
		plt.xlabel('Number of voxels')
		plt.ylabel('Number of segments')
		plt.yscale('log')
		plt.show()
	ids_sorted = np.argsort(-counts)
	unique_ids = unique_ids[ids_sorted]
	counts = counts[ids_sorted]
	return (unique_ids, counts)

#TODO (Jeff): Add get_areas function, should return area vs z