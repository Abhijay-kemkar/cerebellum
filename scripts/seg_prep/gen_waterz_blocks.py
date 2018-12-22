"""
Script to generate blocks of predicted segmentation
"""
import argparse

from cerebellum.data_prep.seg_prep import *

# params from cmd
parser = argparse.ArgumentParser(description='prep and bbox finding of segmentation')
parser.add_argument('-wz', '--wz_thresh', type=float, default=0.5, help='waterz threshold')
parser.add_argument('-i', '--block_index', type=int, default=0, help='block index')
args = parser.parse_args()
wz_thresh = args.wz_thresh
i = args.block_index

###
# SET PARAMS
resolution = (30, 48, 48)
dsmpl = (1,6,6) # from 8 nm
seg8nm_folder = "/n/coxfs01/vcg_connectomics/cerebellum_P7/vol1/waterz/"
bbox = (0, 94, 1246, 1001, 3331, 4176) # global bbox
aff_offset = 14 # affinity offset along z-axis
block_size = 60
###

# load and prepare pred
print "Preparing %dth block for waterz: %.2f"%(i, wz_thresh)
seg8nm_file = seg8nm_folder + "waterz%.2f_pf-align1_1k_crop.h5"%(wz_thresh)
#total_slices = h5py.File(seg8nm_file)["main"].shape[0]
#assert total_slices==bbox[3]-bbox[0]
zz = i*block_size + aff_offset
seg_block_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz)
seg_block = SegPrep(seg_block_name, resolution)
seg_block_lims = ((i*block_size,(i+1)*block_size),(None,None),(None,None))
print seg_block_lims
seg_block.read(seg8nm_file, "main", dsmpl=dsmpl, block_lims=seg_block_lims)
print seg_block.shape
seg_block.write()
# STOP HERE FOR UNFILTERED SEGMENTATION
# Generate bboxes for fiber filtering
# seg_block.gen_bboxes() # Note: Longest step in pipeline
# Assuming bboxes are generated, proceed to filter
