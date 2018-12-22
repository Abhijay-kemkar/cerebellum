"""
Script to run SegPrep on segmentation blocks generated from new affinity networks
"""


from cerebellum.data_prep.seg_prep import *
import argparse

wz_threshes = [0.5]

###
# CMD ARGS
parser = argparse.ArgumentParser(description='run SegPrep on generated segmentation')
parser.add_argument('-i', '--block_index', type=int, default=0, help='block index')
args = parser.parse_args()
i = args.block_index
# SET PARAMS
resolution = (30, 48, 48)
dsmpl = (1,6,6) # from 8 nm
seg8nm_folder = "/n/coxfs01/donglai/srujan/segs/"
block_size = 60
seg_type = '49Ktrain'
###

for wz_id, wz_thresh in enumerate(wz_threshes):
	# load and prepare pred
	print "Preparing %dth block for waterz: %.2f"%(i, wz_thresh)
	zz = i*block_size
	seg8nm_file = seg8nm_folder + "seg_" + seg_type + "_%04d_aff85_his256_%.2f.hdf"%(zz, wz_thresh)
	seg_block_name = "retrained-" + seg_type + "-%04d-wz%.2f"%(zz, wz_thresh)
	seg_block = SegPrep(seg_block_name, resolution)
	seg_block.read(seg8nm_file, "main", dsmpl=dsmpl)
	print seg_block.shape
	seg_block.write()
	# STOP HERE FOR UNFILTERED SEGMENTATION
	# Generate bboxes for fiber filtering
	# seg_block.gen_bboxes() # Note: Longest step in pipeline
	# Assuming bboxes are generated, proceed to filter
