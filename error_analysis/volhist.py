import numpy as np
import h5py
import time
import matplotlib.pyplot as plt

"""
Functions to generate histogram of segment volumes from GT/predicted labels
Lists labels of 20 largest segments
"""

def load_data(label_file):
	"""
	Loads label data from .h5 file into numpy array
	"""
	return np.array(h5py.File(label_file,'r')['main'])

def get_vols(label_data, ids=[], do_save=False, write_file="", do_hist=False):
	"""
	Returns list of tuples (label ID, voxel count) for input ids
	If get_all=True: tuples for all IDs in decreasing order of voxel count
	"""
	if len(ids)==0:
		unique_ids, counts = np.unique(label_data, return_counts=True)
		vols = np.array(sorted(zip(unique_ids, counts), key=lambda tup: tup[1], reverse=True), np.dtype(int,int))
	else:
		counts = [np.count_nonzero(label_data==id) for id in ids]
		vols = np.array(zip(ids, counts), np.dtype(int,int))
	if do_save:
		np.save(write_file,vols)
	if do_hist:
		plt.hist(x=vols[1:,1], bins=100) # remove segment 0
		plt.xlabel('Number of voxels')
		plt.ylabel('Number of segments')
		plt.yscale('log')
		plt.show()
	return vols

if __name__ == "__main__":
	#gt = load_data('/home/srujanm/snemi/zhuhd15_scripts/label.h5')
	pred = load_data('/home/srujanm/snemi/zhuhd15_scripts/tmp_result/withaff85_his256_0.2.h5')

	#gt_vols_file = '/home/srujanm/snemi/analysis_results/gt_all_vols'
	pred_vols_file = '/home/srujanm/snemi/analysis_results/pred_all_vols'
	#top_err_vols_file = '/home/srujanm/snemi/analysis_results/top_err_vols'

	#gt_vols = get_vols(gt, do_hist=True, do_save=True, write_file=gt_vols_file)
	pred_vols = get_vols(pred, do_hist=True, do_save=True, write_file=pred_vols_file)

	#max_arand_labels = np.load('../analysis_results/delta.npy')
	#max_arand_labels = max_arand_labels[:,0].astype(int)

	#top_err_vols = get_vols(gt, ids=max_arand_labels.tolist(), do_save=True, write_file=top_err_vols_file)