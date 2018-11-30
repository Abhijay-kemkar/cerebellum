import os
import json

from cerebellum.ibex.utilities.dataIO import ReadSkeletons
from cerebellum.utils.data_io import *
from skel_methods import *
from voxel_methods import calc_vi

class SkelEval(object):
    """High level methods for error analysis of a segmentation against GT skeletons"""
    def __init__(self, gt_name, pred_name, dsmpl_res=(80,80,80), t_om=0.8, t_m=0.2, t_s=0.7, filtered=False, overwrite_prev=False):
        """
        Args (that are not attributes):
            dsmpl_res (int, int, int): resolution of downsampled GT segmentation before skeletonization
            t_om (float b/w 0 and 1): omission threshold
            t_m (float b/w 0 and 1): merge threshold
            t_s (float b/w 0 and 1): split threshold
            overwrite_prev (bool): overwrite previous results if True
        Attributes:
            gt_name (str): name of GT segmentation
            pred_name (str): name of predicted segmentation
            gt_skeletons (ndarray): list of Skeletons for GT data
            pred (ndarray): predicted segmentation data
            results_folder (str)
            sk_eval (SkeletonEvaluation)
        """
        self.gt_name = gt_name
        self.pred_name = pred_name
        self.gt_skeletons = ReadSkeletons(gt_name, read_edges=True, downsample_resolution=dsmpl_res)
        if not filtered: self.pred = read3d_h5('./segs/' + pred_name + '/seg.h5', 'main')
        if filtered: self.pred = read3d_h5('./segs/' + pred_name + '/filtered-seg.h5', 'main')
        self.results_folder = './err-analysis/' + pred_name
        create_folder(self.results_folder)
        self.sk_eval = None

        if not os.path.exists(self.results_folder+'/skeleton-analysis-summary.json') or overwrite_prev:
            print "Starting error analysis of " + pred_name + " against skeletons of " + gt_name
            self.sk_eval = SkeletonEvaluation(self.pred_name, self.gt_skeletons, self.pred, 
                                                t_om=t_om, t_m=t_m, t_s=t_s, calc_erl=True)
            self.sk_eval.summary(write_path=self.results_folder)
            self.sk_eval.write_errors(write_path=self.results_folder)
            log = open("./logs/" + self.pred_name +'.log', "a+")
            log.write("completed error analysis against skeletons of " + gt_name + "\n\n")
            log.close()
        else:
            print "Skeleton-based error analysis of " + pred_name + " was already completed"

    def get_merges(self, look_in="pred"):
        """
        Returns list of object IDs identified as merges in skeleton-based error analysis
        
        Args:
            look_in (str): if "gt", return GT ids, if "pred", return pred ids
        """
        merge_file = self.results_folder + "/merged-skeletons.ids"
        if not os.path.exists(merge_file):
            print "Error retrieving IDs because skeleton evaluation has not been run yet"
            return
        else:
            f = open(merge_file, "r")
            n_pairs = int(f.readline()) # no of pairs of GT merged skeletons
            if look_in == "pred":
                merge_strs = f.readlines()[::2]
                merge_ids = len(merge_strs)*[0]
                for i, mstr in enumerate(merge_strs):
                    merge_ids[i] = int(mstr)
            elif look_in == "gt":
                merge_strs = f.readlines()[1::2]
                merge_ids = 2*len(merge_strs)*[0]
                for i, mstr in enumerate(merge_strs):
                    merge_ids[2*i] = int(mstr.split(',')[0])
                    merge_ids[2*i+1] = int(mstr.split(',')[1])
            else:
                print "Error: look_in must equal either 'gt' or 'pred'"
                return
            merge_ids = list(set(merge_ids))
            return merge_ids

    def get_corrects(self):
        """Returns list of pred object IDs identified as correct in skeleton-based error analysis"""
        corr_file = self.results_folder + "/correct-skeletons.ids"
        if not os.path.exists(corr_file):
            print "Error retrieving IDs because skeleton evaluation has not been run yet"
            return
        f = open(corr_file, "r")
        n_corr = int(f.readline()) # no of pairs of GT merged skeletons
        corr_ids = f.readlines()
        for i, cstr in enumerate(corr_ids):
            corr_ids[i] = int(cstr.split(',')[1])
        return corr_ids

    def merge_oracle(self):
        """Computes VI after fixing merges from skeleton error analysis"""
        vox_eval = VoxEval(self.gt_name, self.pred_name)
        vi = vox_eval.get_vi()
        print "Original VI split, VI merge: %f, %f"%(vi[0], vi[1])
        fix_ids = self.get_merges(look_in="gt")
        vox_eval.remove_from_gt(vox_eval.missing_objs)
        print "After fixing %d merges in GT flagged by skeleton analysis:"%(len(fix_ids))
        print calc_vi(vox_eval.gt, vox_eval.pred, fix_ids=fix_ids, do_save=True, write_file=self.results_folder+'/vi-merge-oracle.json')