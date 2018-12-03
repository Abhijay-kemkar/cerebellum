"""
Script to generate relabeled GT superblock
"""
from cerebellum.data_prep.seg_prep import *

###
# SET PARAMS
resolution = (30, 48, 48)
#dsmpl = (1,6,6) # from 8 nm
#gt8nm_file = "/home/srujanm/cerebellum/data/vol1/gt8nm_pf-align1_1k_crop.h5"
#bbox = (0, 94, 1246, 1001, 3331, 4176) # global bbox
#file_shape = h5py.File(gt8nm_file)["main"].shape
#assert file_shape == (bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2])
###

gt_superblock_name = "gt-superblock-full"
gt_superblock = SegPrep(gt_superblock_name, resolution)
#gt_superblock.read(gt8nm_file, "main", dsmpl=dsmpl)
gt_superblock.read_internal()
print gt_superblock.shape
#gt_superblock.gen_bboxes()
gt_superblock.read_bboxes()
gt_superblock.relabel(use_bboxes=True, print_labels=False)
gt_superblock.write()