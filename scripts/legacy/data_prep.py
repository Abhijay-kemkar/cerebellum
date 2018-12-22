import json
from cerebellum.utils.seg_prep import *

with open('data_locs.json') as f:
	data_locs = json.load(f)

###########
# prepare 30 nm x 48 nm x 48 nm resolution GT and PF seg blocks
###########

# # prepare GT
# print "Preparing GT"
# resolution = (30, 48, 48)
# gt48nm_file = data_locs["gt"]["dir"] + data_locs["gt"]["48nm"]
# bbox = data_locs["gt"]["48nm-bbox"]
# block_indices = range(2)
# for i in block_indices:
#     print "Generating block %d from"%(i)
#     print gt48nm_file
#     zz = i*data_locs["block-size"]+data_locs["aff-offset"]
#     gt_block = SegPrep("gt%04d"%(zz), resolution)
#     block_lims = ((data_locs["block-size"]*i,data_locs["block-size"]*(i+1)),
#                   (bbox[1],bbox[4]),(bbox[2],bbox[5]))
#     print block_lims
#     gt_block.read(gt48nm_file, "main", block_lims=block_lims)
#     print gt_block.shape
#     gt_block.relabel()
#     gt_block.write()

# # prepare PF segmentation
# print "Prparing PF segmentation cropped to GT extents"
# pf8nm_file = data_locs["initial-seg"]["dir"] + data_locs["initial-seg"]["8nm-pf-linked"]
# dsmpl = (1,1,1)
# offset = data_locs["initial-seg"]["8nm-offset"] # offset aligns segmentation with GT
# for i in block_indices:
#     print "Generating block %d from"%(i)
#     zz = i*data_locs["block-size"]+data_locs["aff-offset"]
#     if zz!=data_locs["aff-offset"]: # adjust block index
#         pf8nm_file = pf8nm_file[:-7]+"%04d.h5"%(zz)
#     print pf8nm_file
#     pf_block = SegPrep("pred-pf-crop2gt-%04d"%(zz), resolution)
#     pf_block_lims = ((0,data_locs["block-size"]),
#                    (max(0,dsmpl[1]*bbox[1]+offset[1]),dsmpl[1]*bbox[4]+offset[1]),
#                    (max(0,dsmpl[2]*bbox[2]+offset[2]),dsmpl[2]*bbox[5]+offset[2]))
#     print pf_block_lims
#     pf_block.read(pf8nm_file, "main", dsmpl=dsmpl, block_lims=pf_block_lims)
#     print pf_block.shape
#     pf_block.padzeros((gt_block.shape[0],gt_block.shape[1]-pf_block.shape[1],gt_block.shape[2]), 1)
#     pf_block.relabel()
#     pf_block.write()

#####################
# prepare 30 nm x 8 nm x 8 nm resolution GT and PF seg blocks
#####################

# prepare PF segmentation
print "Prparing PF segmentation cropped to GT extents"
bbox = data_locs["gt"]["8nm-bbox"]
pf_resolution = (30, 8, 8)
pf8nm_file = data_locs["initial-seg"]["dir"] + data_locs["initial-seg"]["8nm-pf-linked"]
offset = data_locs["initial-seg"]["8nm-offset"] # offset aligns segmentation with GT
block_indices = range(1)
for i in block_indices:
    print "Generating block %d from"%(i)
    zz = i*data_locs["block-size"]+data_locs["aff-offset"]
    if zz!=data_locs["aff-offset"]: # adjust block index
        pf8nm_file = pf8nm_file[:-7]+"%04d.h5"%(zz)
    print pf8nm_file
    pf_block = SegPrep("pred-pf-8nm-crop2gt-%04d"%(zz), pf_resolution)
    pf_block_lims = ((0,data_locs["block-size"]),
                   (max(0,bbox[1]+offset[1]), bbox[4]+offset[1]),
                   (max(0,bbox[2]+offset[2]), bbox[5]+offset[2]))
    print pf_block_lims
    pf_block.read(pf8nm_file, "main", block_lims=pf_block_lims)
    print pf_block.shape
#     pf_block.padzeros(gt_block.shape[1]-pf_block.shape[1], 1)
#     pf_block.padzeros(gt_block.shape[2]-pf_block.shape[2], 2)
#     print pf_block.shape
    pf_block.relabel(id_map=np.load('./segs/pred-pf-crop2gt-%04d/relabeling-map.npy'%(zz)), print_labels=True)
    pf_block.write()