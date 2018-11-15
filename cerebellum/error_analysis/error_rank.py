import numpy as np
import h5py
import time
from em_segLib.seg_eval import CremiEvaluate
from volhist import get_vols

"""
Functions to identify and rank errors in predicted segmentation
"""
#TODO: Make all functions more efficient by adding option for bounding box info

def intersection(seg_gt, seg_p, gt_id):
    """
    Finds all objects in predicted segmentation overlapping with a GT object

    Args:
        seg_gt (ndarray): GT segmentation
        seg_p (ndarray): predicted segmentation
        gt_id (int): GT object of interest
    Returns:
        pred_ids (ndarray), pred_vols (ndarray): IDs of predicted objs, # intersecting voxels
    """
    pred_ids, pred_vols = np.unique(seg_p[seg_gt==gt_id], return_counts=True)
    pred_ids_sorted = np.argsort(-pred_vols) # sort in descending order of intersection
    pred_ids = pred_ids[pred_ids_sorted]
    pred_vols = pred_vols[pred_ids_sorted]
    return (pred_ids, pred_vols)

def pred_fails(seg_gt, seg_p, do_save=False, write_file=""):
    """
    Find all GT objects completely missing in prediction

    Prediction fail occurs when the entire GT object mostly lies in segment 0 of the prediction
    Args:
        seg_gt (ndarray): GT segmentation
        seg_p (ndarray): predicted segmentation
        do_save (bool): flag to save results
        write_file (str): path for result file
    """
    gt_ids, gt_vols = get_vols(seg_gt)
    fails = []
    for i in range(1,len(gt_ids)): # skip segment 0
        pred_ids, pred_vols = intersection(seg_gt, seg_p, gt_ids[i])
        if pred_ids[0]==0:
            fails.append(gt_ids[i])
    if do_save:
        np.save(write_file, fails)
    return np.array(fails)

def union_count(seg_gt, seg_p, gt_id, p_id, return_order=False):
    """
    Returns voxel count in union of a GT object and prediction object
    Args:
        seg_gt (ndarray): GT segmentation
        seg_p (ndarray): predicted segmentation
        gt_id (int): GT object of interest
        p_id (int): pred object of interest
        return_order (bool): True if you wish to return which object is larger
    Returns:
        uc (int)
        order (bool, optional)
    """
    gt_obj = np.ravel_multi_index(np.where(seg_gt==gt_id),seg_gt.shape)
    pred_obj = np.ravel_multi_index(np.where(seg_p==p_id),seg_p.shape)
    uc = len(np.union1d(gt_obj, pred_obj))
    if return_order:
        order = len(gt_obj)>len(pred_obj)
        return uc, order
    else:
        return uc

def IoU_rank(seg_gt, seg_p, skip_ids=[], Ngt=None, Vmin=0, do_save=False, write_file=""):
    """
    Ranks GT segments in decreasing order of prediction error

    Computes IoU between each GT segment and best overlapped predicted segment
    Args:
        seg_gt (ndarray): GT segmentation
        seg_p (ndarray): predicted segmentation
        skip_ids (ndarray): list of GT ids to skip (in addition to 0)
        Ngt (int): calculate IoU rank for Ngt largest segments in GT
        Vmin (int): min volume of segments to be considered in # voxels
        do_save (bool): flag to save results
        write_file (str): path for result file
    Returns:
        id_calc (ndarray), iou (ndarray), bestpred (ndarray): GT ids, IoU, best prediction ID
    """
    # get GT segments
    seg_gt_id, seg_gt_cc = get_vols(seg_gt)
    seg_gt_id = seg_gt_id[1:] # delete segment 0
    seg_gt_cc = seg_gt_cc[1:]
    # restrict based on input conditions
    if Ngt is not None:
        seg_gt_id = seg_gt_id[:Ngt]
        seg_gt_cc = seg_gt_cc[:Ngt] 
    id_calc = seg_gt_id[seg_gt_cc > Vmin]
    for i in skip_ids:
        id_calc = np.delete(id_calc, np.where(id_calc==i))
    num_calc = len(id_calc)
    print "Calculating IoU scores for %d objects in GT" %(len(id_calc))
    iou = np.zeros(num_calc)
    best_pred = np.zeros(num_calc)
    for i in range(num_calc):
        pred_ids, pred_vols = intersection(seg_gt, seg_p, id_calc[i])
        # remove background voxels from intersection
        pred_ids = pred_ids[np.nonzero(pred_ids)]
        pred_vols = pred_vols[np.nonzero(pred_ids)]
        best_pred[i] = pred_ids[0]
        iou[i] = float(pred_vols[0])/union_count(seg_gt,seg_p,id_calc[i],pred_ids[0]) # IoU with best overlapping object
        numV = float(sum(pred_vols))
        print id_calc[i]
    # sort GT objects in ascending order of IoU score
    iou_sort = np.argsort(iou)
    id_calc = id_calc[iou_sort]
    iou = iou[iou_sort]
    best_pred = best_pred[iou_sort]
    if do_save:
        results = np.vstack((id_calc, iou, best_pred))
        np.save(write_file, results)
    return id_calc, iou, best_pred

def slice_iou(last_slice, first_slice):
    """
    Generates IoU scores of all objects across two slices
    
    Args:
        last_slice (ndarray 1 x X x Y): objects in this slice are compared
        first_slice (ndarray 1 x X x Y): against objects in this slice
    """
    n_objs = np.max(last_slice)
    ints = np.zeros(n_objs)
    unions = np.zeros(n_objs)
    orders = np.zeros(n_objs)
    ious = np.zeros(n_objs)
    for i in range(n_objs):
        if len(np.flatnonzero(last_slice==i))==0: # no voxels of this object in the slice
            continue
        front_ids, front_vols = intersection(last_slice, first_slice, i)
        ints[i] = front_vols[0]
        unions[i], orders[i] = union_count(last_slice, first_slice, i, front_ids[0], return_order=True)
        ious[i] = float(ints[i])/unions[i]
    return ints, unions, ious, orders

def calc_vi(seg_gt, seg_p, fix_ids=None):
    """
    Calculates VI between two segmentations
    
    Args:
        seg_gt (ndarray): GT segmentation
        seg_p (ndarray): predicted segmentation
        fix_ids (list of ints): apply oracle to these IDs in GT segmentation
    Returns:
        vi_split_oracle, vi_merge_oracle
    """
    seg_oracle = seg_p.copy()
    if fix_ids is not None:
        max_id = np.max(seg_p) + 1
        for i in range(len(fix_ids)):
            seg_oracle[seg_gt==fix_ids[i]] = max_id # oracle for this GT object
            max_id += 1
    vi_split_oracle, vi_merge_oracle = CremiEvaluate(seg_oracle.astype(np.int), seg_gt.astype(np.int))
    del seg_oracle
    return vi_split_oracle, vi_merge_oracle

def vi_rank(seg_gt, seg_p, iou_results, iou_max=0.7, do_save=True, write_file=""):
    """
    Ranks GT segments in order of their contribution to VI error

    Calculates deltaVI by applying an oracle for each segment
    VI calculation function is CremiEvaluate
    Args:
        seg_gt (ndarray): GT segmentation
        seg_p (ndarray): predicted segmentation
        iou_results (ndarray): results from IoU_rank function
        iou_max (float): ignore GT segments whose IoU score is higher than this
        do_save (bool): flag to save results
        write_file (str): path for result file
    Returns:
        id_calc (nx, ndarray), deltaVI (nx2 ndarray): 
    """
    gt_ids, iou_scores = iou_results[0,:], iou_results[1,:]
    vi_split, vi_merge = CremiEvaluate(seg_p.astype(np.int), seg_gt.astype(np.int))
    print 'VI-split: %1.5f' %(vi_split)
    print 'VI-merge: %1.5f' %(vi_merge)
    id_calc = []
    deltaVI = []
    max_id = np.max(seg_gt) + 1
    for i in range(len(gt_ids)):
        if iou_scores[i] > iou_max:
            continue
        seg_oracle = seg_p.copy()
        seg_oracle[seg_gt==gt_ids[i]] = max_id # oracle for this GT object
        vi_split_oracle, vi_merge_oracle = CremiEvaluate(seg_oracle.astype(np.int), seg_gt.astype(np.int)) 
        delta_vi = [vi_split-vi_split_oracle, vi_merge-vi_merge_oracle]
        print 'Label %9d in GT --> delta VI split %1.5f, delta VI merge %1.5f' %(gt_ids[i],delta_vi[0],delta_vi[1])
        id_calc.append(gt_ids[i])
        deltaVI.append(delta_vi)
        del seg_oracle
    id_calc = np.array(id_calc)
    deltaVI = np.array(deltaVI)
    deltaVI_tot = deltaVI[:,0] + deltaVI[:,1] # sum of split and merge
    sortbyVI = np.argsort(-deltaVI_tot)
    id_calc = id_calc[sortbyVI] # sort in descending order of error
    deltaVI = deltaVI[sortbyVI,:]
    if do_save:
        results = np.vstack((id_calc, deltaVI.transpose()))
        np.save(write_file, results)
    return id_calc, deltaVI