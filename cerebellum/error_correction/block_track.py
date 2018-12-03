from cerebellum.error_correction.slice_stitch import slice2slice_iou_calc
import time
import numpy as np

def block_track(block_seg, iou_thresh, zstride=1):
    """
    Naive slice to slice tracking without using object bboxes
    """
    total_time = 0
    if zstride>0: track_range = range(0,block_seg.shape[0]-1,zstride)
    elif zstride<0: track_range = range(block_seg.shape[0],1,zstride)
    for sslice_id in track_range:
        print "On slice: ", sslice_id
        slice_time = time.time()
        tslice_id = sslice_id + zstride
        sslice = np.array([block_seg[sslice_id,:,:]])
        tslice = np.array([block_seg[tslice_id,:,:]])
        # if bboxes are unavailable
        s_objs = np.unique(sslice)
        for s_obj in s_objs.tolist():
            int_ids, ints, unions = slice2slice_iou_calc(sslice, tslice, s_obj)
            t_obj = int_ids[0]
            iou = float(ints[0])/unions[0]
            if iou>iou_thresh:
                block_seg[block_seg==t_obj] = s_obj
        total_time += time.time()-slice_time
    print "Runtime: %f"%(total_time)
    return block_seg