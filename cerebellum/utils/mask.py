from skimage.measure import regionprops
import numpy as np

"""
Functions to generate object masks
Args:
	seg (array): segmentation array
	obj_id (int): id of object of interest
"""

def gen_mask(seg, obj_id):
	obj_mask = np.zeros_like(seg, dtype=int)
	obj_mask[seg==obj_id] = 1
	return obj_mask

def get_bbox(seg, obj_id):
	regions = regionprops(gen_mask(seg, obj_id))
	return regions[0].bbox

def mask_in_bbox(seg, obj_id):
	obj_mask = gen_mask(seg, obj_id)
	regions = regionprops(obj_mask)
	bbox = regions[0].bbox
	return obj_mask[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]], bbox