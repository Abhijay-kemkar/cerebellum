import numpy as np
import time

from cerebellum.error_analysis.voxel_methods import *

"""
Functions to link split objects via slice to slice tracking
"""

def slice2slice_iou_calc(source_slice, target_slice, source_id, search_lims=None):
    """
    Returns list of tracked id in target slice for source objects and IoU score

    Args:
        source_slice (nxm array)
        target_slice (nxm array): same size as source slice
    """
    assert source_slice.shape==target_slice.shape
    #print np.array([source_slice]).shape
    int_ids, int_vols = intersection_list(np.array([source_slice]), np.array([target_slice]), source_id)
    union_vols = np.zeros_like(int_vols)
    for i in range(len(int_ids)):
        union_vols[i] = union_count(np.array([source_slice]), np.array([target_slice]), source_id, int_ids[i])
    return int_ids, int_vols, union_vols