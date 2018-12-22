"""
Script to generate GT blocks of chosen size with good labels
"""
import argparse
from cerebellum.data_prep.seg_prep import *

# params from cmd                                                                           
parser = argparse.ArgumentParser(description='prep GT segmentation blocks')       
parser.add_argument('-i', '--block_index', type=int, default=0, help='block index')
args = parser.parse_args()                                                                  
i = args.block_index

###
# SET PARAMS
resolution = (30, 48, 48)
dsmpl = (1,6,6) # from 8 nm
gt8nm_file = "/n/coxfs01/vcg_connectomics/cerebellum_P7/vol1/gt8nm_pf-align1_1k_crop.h5"
bbox = (0, 94, 1246, 1001, 3331, 4176) # global bbox
block_size = 60
aff_offset = 14
total_slices = h5py.File(gt8nm_file)["main"].shape[0]
assert total_slices==bbox[3]-bbox[0]
###

print "Generating GT block %d from"%(i)
print gt8nm_file
zz = i*block_size+aff_offset
gt_block_name = "gt-48nm-%dslices-%04d"%(block_size, zz)
gt_block = SegPrep(gt_block_name, resolution)
block_lims = ((block_size*i, block_size*(i+1)),(None,None),(None,None))
print block_lims
gt_block.read(gt8nm_file, "main", dsmpl=dsmpl, block_lims=block_lims)
print gt_block.shape
gt_block.gen_bboxes()
#gt_block.read_bboxes()
gt_block.relabel(use_bboxes=True, print_labels=False)
gt_block.write()
