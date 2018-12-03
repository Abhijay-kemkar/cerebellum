from cerebellum.error_analysis.skel_segeval import *

wz_threshes = [0.5]

###
# SET PARAMS
with open('data_locs.json') as f:
	data_locs = json.load(f)
resolution = (30, 48, 48)
dsmpl = (1,6,6) # from 8 nm
seg8nm_folder = "/home/srujanm/cerebellum/data/vol1/waterz/"
bbox = data_locs["gt"]["8nm-bbox"] # global bbox
aff_offset = data_locs["aff-offset"] # affinity offset along z-axis
block_size = 60
# set skeleton error analysis thresholds
t_om = 0.9
t_m = 0.5
t_s = 0.8
###

for wz_id, wz_thresh in enumerate(wz_threshes):
	i = 0 # block index - iterable if chosen
	zz = i*block_size + aff_offset
	gt_block_name = "gt-48nm-%dslices-%04d"%(block_size, zz)
	seg_block_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz)
	# evaluate unfiltered pred against GT skeletons
	# results are saved to ./err-analysis/<seg_block_name>/folder
	vox_eval = VoxEval(gt_block_name, seg_block_name, stage=None)
	vox_eval.run_fullsuite(iou_max=0.6, hist_segs=10, overwrite_prev=False)
	skel_eval = SkelEval(gt_block_name, seg_block_name, dsmpl_res=(80,80,80), 
	                     t_om=t_om, t_m=t_m, t_s=t_s, 
	                     include_zero_split=False, include_zero_merge=True,
	                     stage=None, overwrite_prev=True)
	skel_eval.merge_oracle()
	skel_eval.split_oracle()

	# repeat error analysis for filtered objects
	# results are saved to ./err-analysis/<seg_block_name>/folder
	vox_eval = VoxEval(gt_block_name, seg_block_name, stage="filtered")
	vox_eval.run_fullsuite(iou_max=0.6, hist_segs=10, thresh_miss=t_om, overwrite_prev=False)
	skel_eval = SkelEval(gt_block_name, seg_block_name, dsmpl_res=(80,80,80), 
	                     t_om=t_om, t_m=t_m, t_s=t_s, 
	                     include_zero_split=False, include_zero_merge=True,
	                     stage="filtered", overwrite_prev=True)
	skel_eval.merge_oracle()
	skel_eval.split_oracle()