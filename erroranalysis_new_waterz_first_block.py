from cerebellum.error_analysis.skel_segeval import *

#wz_threshes = [0.3, 0.4, 0.5, 0.6] # waterz thresholds
wz_threshes = [0.5] # waterz thresholds

###
# CMD ARGS
parser = argparse.ArgumentParser(description='run error analysis on generated segmentation')
parser.add_argument('-i', '--block_index', type=int, default=0, help='block index')
args = parser.parse_args()
i = args.block_index
# SET PARAMS
resolution = (30, 48, 48)
block_size = 60 # number of slices along z-axis
aff_offset = 14
seg_type = 'train49K'
# set skeleton error analysis thresholds
t_om = 0.9
t_m = 0.5
t_s = 0.8
###

for wz_id, wz_thresh in enumerate(wz_threshes):
	zz_gt = i*block_size + aff_offset
	zz_seg = i*block_size
	gt_block_name = "gt-48nm-%dslices-%04d"%(block_size, zz_gt)
	seg_block_name = "retrained-"+seg_type+"-%04d-wz%.2f"%(zz_seg, wz_thresh)
	# evaluate unfiltered pred against GT skeletons
	# results are saved to ./err-analysis/<seg_block_name>/folder
	print "Evaluating unfiltered segmentation"
	vox_eval = VoxEval(gt_block_name, seg_block_name, stage=None)
	#vox_eval.run_fullsuite(iou_max=0.6, hist_segs=10, overwrite_prev=True)
	vox_eval.find_misses(thresh_miss=t_om)
	vox_eval.find_vi()
	skel_eval = SkelEval(gt_block_name, seg_block_name, dsmpl_res=(80,80,80), 
	                     t_om=t_om, t_m=t_m, t_s=t_s, 
	                     include_zero_split=False, include_zero_merge=True,
	                     stage=None, overwrite_prev=True)
	#skel_eval.merge_oracle()
	#skel_eval.split_oracle()