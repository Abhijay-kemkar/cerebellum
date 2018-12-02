import numpy as np
import time

from cerebellum.error_correction.slice_stitch import slice2slice_iou_calc

"""
Functions to propagate labels from block to block
"""

def block_lock(sblock_seg, tblock_seg, iou_thresh=0.5, 
               sbbox_dict=None, tbbox_dict=None, search_span=None):
    """
    Locks objects in target block to IDs of objects in source block
    
    Tracks objects in last slice of source block into first slice of target block
    
    Currntly only supports sbbox_dict = None option
    """
    start_time = time.time()
    for i in [1,2]:
        assert sblock_seg.shape[i]==tblock_seg.shape[i]
    sslice = np.array([sblock_seg[-1,:,:]])
    tslice = np.array([tblock_seg[0,:,:]])
    # if bboxes are unavailable
    s_objs = np.unique(sslice)
    t_objs = np.zeros_like(s_objs, dtype=np.uint32)
    for s_obj in s_objs:
        if sbbox_dict is None:
            int_ids, ints, unions = slice2slice_iou_calc(sslice, tslice, s_obj)
        else:
            sbbox = sbbox_dict[s_obj]
            s_size = (search_span*(sbbox[4]-sbbox[1]),
                      search_span*(sbbox[5]-sbbox[2]))
            cropped_sslice = sslice[:,
                                   max(0,sbbox[1]-s_size[0]):
                                   min(sslice.shape[1],sbbox[4]+s_size[0]),
                                   max(0,sbbox[2]-s_size[1]):
                                   min(sslice.shape[2],sbbox[5]+s_size[1])]
            cropped_tslice = tslice[:,
                                   max(0,sbbox[1]-s_size[0]):
                                   min(sslice.shape[1],sbbox[4]+s_size[0]),
                                   max(0,sbbox[2]-s_size[1]):
                                   min(sslice.shape[2],sbbox[5]+s_size[1])]
            int_ids, ints, unions = slice2slice_iou_calc(cropped_sslice, 
                                                         cropped_tslice, s_obj)
        t_obj = int_ids[0]
        iou = float(ints[0])/unions[0]
        if iou>iou_thresh:
            if tbbox_dict is None:
                tblock_seg[tblock_seg==t_obj] = s_obj
            else:
                tbbox = tbbox_dict[t_obj]
                cropped_tblock = tblock_seg[tbbox[0]:tbbox[3],
                                            tbbox[1]:tbbox[4],
                                            tbbox[2]:tbbox[5]]
                change_vox = list(np.nonzero(cropped_tblock==t_obj))
                change_vox = tuple([cv + tbbox[c_id] for c_id, cv in enumerate(change_vox)])
                tblock_seg[change_vox] = s_obj
    print "Runtime: %f"%(time.time()-start_time)
    return tblock_seg