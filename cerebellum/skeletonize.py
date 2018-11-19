from utils.data_io import *
from ibex.transforms.seg2seg import DownsampleMapping
from ibex.skeletonization.generate_skeletons import TopologicalThinning, FindEndpointVectors, FindEdges

def check_ids(seg):
	"""
	checks if all IDs in segmentation are consecutive
	Args:
		seg (ndarray): segmentation data
	"""
	seg_ids = np.unique(seg)
	max_id = np.max(seg_ids)
	n_ids = len(seg_ids)
	try:
	    assert max_id == n_ids-1
	except:
	    missing_ids = np.sort(np.array(list(set(range(max_id+1)).difference(set(seg_ids)))))
	    print "Error! Labels in segmentation are not consecutive. %d IDs are missing"%(len(missing_ids))
	    print missing_ids

def gen_skeletons(seg_name, dsmpl_res=(80,80,80)):
	"""
	Function to skeletonize a segmentation
	Args:
		seg_name (str): name of prepped segmentation
		dsmpl_res (int, int, int): resolution to downsample segmentation to before skeletonziation in nm
	"""
	seg = read3d_h5('./segs/' + seg_name + '/seg.h5', 'main')
	seg = seg.astype(np.int64)
	check_ids(seg)
	create_folder('./skeletons/')
	DownsampleMapping(seg_name, seg, output_resolution=dsmpl_res)
	TopologicalThinning(seg_name, seg, skeleton_resolution=dsmpl_res)
	FindEndpointVectors(seg_name, skeleton_resolution=dsmpl_res)
	FindEdges(seg_name, skeleton_resolution=dsmpl_res)