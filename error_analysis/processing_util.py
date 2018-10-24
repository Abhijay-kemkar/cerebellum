import h5py
import numpy as np
from T_util import writeh5_bk
import cv2
import em_segLib
from skimage import measure, morphology
from scipy.ndimage.morphology import distance_transform_edt

def eval(pred,gt=None,dilate=False):
    if gt is None:
        gt = np.array(h5py.File('/home/zhuhd15/Lab/snemi/data/test_labels.h5','r')['main'])
    if dilate:
        gt = em_segLib.seg_dist.DilateData(gt,2)
    return em_segLib.seg_eval.adapted_rand(pred.astype(np.int),gt.astype(np.int))

def remove_singleton(data=None, fetch=None, save=None):
    if data is None:
        data = np.array(h5py.File(fetch,'r')['main'])
    for z in range(data.shape[0]):
        for label in np.unique(data[z]):
            neighbour = False
            if z > 0:
                if np.sum(data[z-1]==label):
                    neighbour = True
            if z < data.shape[0] - 1:
                if np.sum(data[z+1]==label):
                    neighbour = True
        if not neighbour:
            data[z][data[z]==label] == 0
    if save is not None:
        writeh5_bk(save, 'main', data)
    return data

def erode_fragments(data=None,fetch=None,save=None,pixel_id=None,rep=10,erode_size=10,dilate_size=10):
    if data is None:
        data = np.array(h5py.File(fetch,'r')['main'])
    erode_kernel = np.ones((erode_size,erode_size),np.uint8)
    dilate_kernel = np.ones((dilate_size,dilate_size),np.uint8) 
    for z in range(data.shape[0]):
        bd_map = (data[z] == 0)         # find places for boundarys
        pix_sets = [pix_id] if pix_id is not None else np.unique(data[z])
        for pix in pix_sets:
            if pix == 0 and len(pix_sets) > 1:
                continue
            pix_map = (data[z] == pix_id).astype(np.uint8)
            pix_cpy = np.copy(pix_cpy)
            if np.sum(pix_map) == 0:
                continue
            for _ in range(rep):
                pix_map = cv2.dilate(pix_map,iterations=1,kernel=dilate_kernel)
                pix_map[(1-bd_map).astype(np.bool)] = 0                             # mask un-needed areas
                pix_map[img == pix] = 1
                pix_map = cv2.erode(pix_map,iterations=1,kernel=erode_kernel)
                pix_map = cv2.dilate(pix_map,iterations=1,kernel=dilate_kernel)
            if len(pix_sets) == 1 and pix_id == 0:
                data[z][pix_map.astype(bool) * bd_map] = pix
            else:
                data[z][pix_cpy.astype(bool) * (1-pix_map).astype(bool)] = 0
    if save is not None:
        writeh5_bk(save, 'main', data)
    return data

def missing_z_fix(data=None,fetch=None,save=None,threshold=0.5,erode=False,pix_id=None):
    if data is None:
        data = np.array(h5py.File(fetch,'r')['main'])
    for z in range(data.shape[0]):
        if z > 1:
            data[z] = z_single_layer_fix(data[z],data[z-1],pix_id=pix_id,threshold=threshold,erode=erode)
        if z < data.shape[0] - 1:
            data[z] = z_single_layer_fix(data[z],data[z+1],pix_id=pix_id,threshold=threshold,erode=erode)
    if save is not None:
        writeh5_bk(save, 'main', data)
    return data

def tracking_filling(data=None,fetch=None,save=None):
    if data is None:
        data = np.array(h5py.File(fetch,'r')['main'])
    for z in range(1,data.shape[0]-1):
        boundary = data[z] == 0
        xs,ys = np.where(boundary)
        for i in range(xs.size):
            if data[z-1,xs[i],ys[i]]==data[z+1,xs[i],ys[i]] and data[z-1,xs[i],ys[i]] != 0: 
                data[z,xs[i],ys[i]] = data[z-1,xs[i],ys[i]]
    if save is not None:
        writeh5_bk(save, 'main', data)
    return data

def z_single_layer_fix(layer_fix,layer_compare,threshold=0.5,pix_id=None,erode=False):
    connect,max_label = measure.label(layer==0, neighbors=4,return_num=True)
    for label in range(1,max_label+1):
        pix_id_set = [pix_id] if pix_id is not None else np.unique(layer_compare[connect ==label])
        for pix_label in pix_id_set:
            if np.sum((layer_compare==pix_label) * (connect ==label)) > np.sum(connect==label)*threshold:
                color_map = connect == label
                if erode:
                    color_map = cv2.erode(color_map.astype(np.uint8),iterations=1,kernel=np.ones((erode,erode),np.uint8))
                    color_map = cv2.dilate(color_map,iterations=1,kernel=np.ones((erode,erode),np.uint8))
                    color_map = (color_map.astype(np.bool) * (connect==label))
                layer_fix[color_map] = pix_label
    return layer_fix

def fix_boundary(data=None,fetch=None,save=None,affinity=None,pixel_id=None,distance=20):
    if data is None:
        data = np.array(h5py.File(fetch,'r')['main'])
    affinity = np.array(h5py.File(affinity,'r')['main'])[0]
    print 'load successfully'
    for z in range(data.shape[0]):
        print 'layer ',z
        if not np.sum(data[z]==pixel_id):
            continue
        label_area = 1-(data[z]==pixel_id)
        label_area = (distance_transform_edt(label_area) < distance)
        label_area = label_area * (affinity[z] < 0.5)
        data[z][label_area] = pixel_id
        '''ids = np.unique(label_area * data[z])
        for label_id in ids:
            if label_id == 0:
                continue
            print label_id,np.sum(data[z]==label_id)
            if np.sum(data[z]==label_id) < 500:
                data[z][data[z]==label_id] = pixel_id
        '''
        pass
    if save is not None:
        writeh5_bk(save, 'main', data)
    return data

def pred_analysis(seg_p,seg_gt):
    seg_gt_id, seg_gt_cc = np.unique(seg_gt, return_counts=True)
    seg_gt_id = np.delete(seg_gt_id,0)
    seg_gt_cc = np.delete(seg_gt_cc,0)
    sid = np.argsort(-seg_gt_cc)
    seg_gt_id = seg_gt_id[sid]
    seg_gt_cc = seg_gt_cc[sid]

    # big seg only
    ind = seg_gt_id[seg_gt_cc>1000]
    num = len(ind)
    print 'Total Len:',num
    iou = np.zeros(num)
    IoU_data = []
    for i in range(num):
        tmp_id, tmp_cc = np.unique(seg_p[seg_gt==ind[i]], return_counts=True)
        tmp_sid = np.argsort(-tmp_cc)
        iou[i] = float(max(tmp_cc))/len(np.union1d(np.ravel_multi_index(np.where(seg_gt==ind[i]),seg_gt.shape), np.ravel_multi_index(np.where(seg_p==tmp_id[np.argmax(tmp_cc)]),seg_p.shape)))
        numV = float(sum(tmp_cc))
        #tmp_IoU = []
        IoU_data.append([ind[i],numV,iou[i]])
        #print 'Total length:', num
        if i % int(num/10) == 0:
            print 'Processed %2.2f' %(i * 100 /num)
    return np.array(IoU_data)

from em_segLib.seg_eval import CremiEvaluate

def gt_evaluation(IoU_set,pred,gt,seg_gt,maxT=3600):
    import em_segLib
    import time
    #from em_segLib.seg_dist import DilateData
    IoU_set = IoU_set[(-IoU_set[:,1]).argsort()] # sort GT objects in descending order of their IoU with corresp pred object
    #print IoU_set
    #seg_gt = DilateData(gt,6)
    gt_id,area,IoU = IoU_set[:,0],IoU_set[:,1],IoU_set[:,2]
    vi = CremiEvaluate(pred.astype(np.int), seg_gt.astype(np.int)) # VI scores
    start = time.time()
    print 'current VI-split: %1.5f' %(vi[0])
    print 'current VI-merge: %1.5f' %(vi[1])
    max_id = np.max(gt) + 1
    diff_list = []
    for i in range(len(gt_id)):
        if area[i] < 5000 or time.time() - start > maxT:
            break
        if IoU[i] > 0.7: # ignore objects predicted with high IoU
            continue
        tpred = pred.copy()
        tpred[gt==gt_id[i]] = max_id # oracle for this GT object
        vi_oracle = CremiEvaluate(tpred.astype(np.int), seg_gt.astype(np.int)) # VI scores after oracle
        print 'Time remaining:%5d'%(maxT-time.time()+start)
        print 'Label: %9d generates delta VI split %1.5f, delta VI merge %1.5f' %(gt_id[i],vi[0]-vi_oracle[0],vi[1]-vi_oracle[1])
        diff_list.append([gt_id[i],vi[0]-vi_oracle[0],vi[1]-vi_oracle[1]])
        del tpred
    diff_list = np.array(diff_list)
    return diff_list

def chull_fix(data=None,fetch=None,save=None,pixel_id=None,axis=0,ignore_pixel=False):
    if data is None:
        data = np.array(h5py.File(fetch,'r')['main'])
    for z in range(data.shape[axis]):
        if axis == 0:
            pass
    #TODO : finish convex_hull function
