from cerebellum.data_prep.seg_prep import *
from cerebellum.error_analysis.skel_segeval import *
import json

###
resolution = (30, 48, 48)
block_size = 60
affinity_offset = 14
wz_thresh = 0.5
# error analysis params
t_om = 0.9
t_m = 0.5
t_s = 0.8
###
gt_superblock_type = 'gt48nm-linear-superblock'
pred_superblock_type = 'waterz%.2f-48nm-crop2gt-linear-superblock'%(wz_thresh)

superblock_steps = [1, 2, 4, 8, 12, 16]
total_time = 0
for n_blocks in superblock_steps:
	# prep GT superblock for error analysis
	block_time = time.time()
	gt_superblock_name = gt_superblock_type + '-%dblocks'%(n_blocks)
	pred_superblock_name = pred_superblock_type + '-%dblocks'%(n_blocks)
	print "Preparing %d blocks superblock for error analysis"%(n_blocks)
	# prep pred_superblock for error analysis
	pred_superblock_seg = SegPrep(pred_superblock_name, resolution)
	pred_superblock_seg.read('./superblocks/'+pred_superblock_name+'.h5', 'main')
	pred_superblock_seg.write()
	print "Finished saving pred superblock segmentation"
	# run error analysis
	superblock_vox_eval = VoxEval(gt_superblock_name, pred_superblock_name)
	superblock_vox_eval.find_misses(thresh_miss=t_om)
	superblock_vox_eval.find_vi()
	superblock_skel_eval = SkelEval(gt_superblock_name, pred_superblock_name, dsmpl_res=(80,80,80), 
	                                 t_om=t_om, t_m=t_m, t_s=t_s, 
	                                 include_zero_split=False, include_zero_merge=True,
	                                 overwrite_prev=True)
	print "Error analysis for %d blocks superblock complete in %d s"%(n_blocks, time.time()-block_time)
	total_time += time.time()-block_time
	#superblock_skel_eval.merge_oracle()
	#superblock_skel_eval.split_oracle()
print "Total superblock error scaling analysis time: %d s"%(total_time)