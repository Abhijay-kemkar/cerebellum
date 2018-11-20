import json

from cerebellum.error_analysis.skel_segeval import *

with open('data_locs.json') as f:
	data_locs = json.load(f)

block_indices = range(1)

for i in block_indices:
	zz = i*data_locs["block-size"]+data_locs["aff-offset"]
	gt_name = "gt%04d"%(zz)
	pred_name = "pred-pf-crop2gt-%04d"%(zz)
	skel_eval = SkelEval(gt_name, pred_name, dsmpl_res=(80,80,80), t_om=0.8, t_m=0.2, t_s=0.7, overwrite_prev=False)
	skel_eval.merge_oracle()