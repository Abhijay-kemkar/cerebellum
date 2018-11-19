import json

from cerebellum.error_analysis.voxel_segeval import *

with open('data_locs.json') as f:
	data_locs = json.load(f)

block_indices = range(1)

for i in block_indices:
	zz = i*data_locs["block-size"]+data_locs["aff-offset"]
	gt_name = "gt%04d"%(zz)
	pred_name = "pred-pf-crop2gt-%04d"%(zz)
	print "Starting voxel evaluation methods for " + pred_name + " against " + gt_name
	vox_eval = VoxEval(gt_name, pred_name)
	vox_eval.find_misses()
	iou_thresh = 0.6
	vox_eval.find_ious(print_thresh=iou_thresh)
	vox_eval.find_delta_vis(iou_max=iou_thresh)