import numpy as np
import time
import json

from id_map import IDmap
from cerebellum.stats.voxel_stats import get_vols
from voxel_segeval import VoxEval

class NodeAssignment:
    def __init__(self, gt_skeleton, pred):
        """
        Assignment of nodes of a GT skeleton to objects in predicted segmentation
        
        Args:
            gt_skeleton (Skeleton)
            pred (ndarray): predicted segmentation
        Attributes:
            label (str): ID of GT skeleton
            n_nodes (int): no. of edges
            objs (ndarray): array of pred labels for nodes
        """
        self.label = gt_skeleton.label
        nodes = gt_skeleton.get_nodes()
        n_nodes = nodes.shape[0]
        self.n_nodes = n_nodes
        self.objs = np.zeros(n_nodes)
        for i in range(n_nodes):
            self.objs[i] = pred[tuple(nodes[i,:])]
    
    def get_id_maps(self):
        """Returns ID map from GT skeleton ID -> intersecting pred objects"""
        pred_ids, pred_counts = np.unique(self.objs, return_counts=True)
        size_order = np.argsort(-pred_counts)
        pred_ids = pred_ids[size_order].astype(int).tolist() # sort in descending order of intersection
        pred_counts = pred_counts[size_order].tolist()
        return IDmap(self.label, pred_ids, size_in=self.n_nodes, counts_out=pred_counts)

class SkeletonEvaluation:
    # TODO: add ERL method for split errors based on CRCs of skeleton
    def __init__(self, title, gt_skeletons, pred, t_om=0.8, t_m=0.5, t_s=0.7, calc_erl=False):
        """
        Evaluation of a predicted segmentation w.r.t. GT skeletons
        
        Args:
            title (str)
            gt_skeletons (list of Skeletons): output of ReadSkeletons function
            pred (ndarray): predicted segmentation
            t_om (float b/w 0 and 1): omission threshold
                A GT skeleton is an omission if a fraction >=t_om of its nodes lie in predicted segment 0.
                Lower t_om -> more omissions detcted
            t_m (float b/w 0 and 1): merge threshold
                A GT skeleton S is a merge if there exists another GT skeleton T such that a 
                fraction >=t_m of the nodes of T have the same pred label as fraction >=t_m of the nodes of S.
                Lower t_m -> more merges detected
            t_s (float b/w 0 and 1): split threshold
                A GT skeleton S is a split if its top predicted label covers <=t_s of its nodes.
                Lower t_s -> more splits detected
        Attributes:
            title (str)
            n_skels (int): number of skeletons
            grid_size (int, int, int): size of predicted segmentation in voxels
            t_om (float)
            t_m (float)
            t_s (float)
            categories (list of n_skels strings):
                each element is "omitted", "split", "merged", "hybrid" or "correct"
            erl_pred (float): expected run length of predicted segmentation in nm
            erl_gt (float): expected run length of GT skeletons in nm
            merge_list (list of ID maps): each ID map from pred ID -> merged GT ids
            split_list (list of ID maps): each ID map from GT id -> split pred ids
            corr_list (list of ID maps): each ID map from GT id -> pred id
            gt2pred (list of ID maps): of all objects from GT id -> pred ids
        """
        def merge_check(id_map1, id_map2, t_m=t_m):
            """
            Used to check if two GT skeletons have a merge in the predicted segmentation
            
            Args:
                id_map1, id_map2: ID maps of GT skeletons -> pred objects
            Returns:
                merged_ids (list of ints): IDs of predicted objects corresponding to skeleton merge
            """
            top_ids1 = [id_map1.ids_out[i] for i in range(len(id_map1.ids_out)) 
                        if (1.*id_map1.counts_out[i])/id_map1.size_in > t_m]
            top_ids2 = [id_map2.ids_out[i] for i in range(len(id_map2.ids_out)) 
                        if (1.*id_map2.counts_out[i])/id_map2.size_in > t_m]
            merged_ids = list(set(top_ids1) & set(top_ids2))
            # ignore if both objects are mostly in 0, this could be an unlabeled object in GT
            if 0 in merged_ids:
                merged_ids.remove(0)
            if len(merged_ids)==0:
                return []
            else:
                return merged_ids
            
        def split_check(skeleton, id_map, t_s=t_s):
            """
            Used to check if a GT skeleton is a split in the predicted segmentation
            """
            assert skeleton.label == id_map.id_in
            pred_ids = id_map.ids_out
            # ignore splits that involve purely 0
            if 0 in pred_ids:
                pred_ids.remove(0)
            if len(pred_ids) == 1 or (1.*pred_ids[0])/id_map.size_in > t_s:
                return [pred_ids[0]]
            else:
                return pred_ids
        
        start_time = time.time()
        self.title = title
        self.n_skels = len(gt_skeletons)
        try:
            pred_ids,_ = get_vols(pred)
            max_pred = np.max(pred)
            assert len(pred_ids) == max_pred+1
        except:
            print "Error! Labels in predicted segmentation are not consecutive"
        self.n_preds = len(pred_ids)
        try:
            assert gt_skeletons[0].grid_size == pred.shape
        except:
            print "Predicted segmentation and GT skeleton volume have different grid size"
        self.grid_size = (pred.shape[0], pred.shape[1], pred.shape[2])
        print 'Starting evaluation of %d labels in %dx%dx%d predicted segmentation against %d GT skeletons'%(self.n_preds, pred.shape[0], pred.shape[1], pred.shape[2], self.n_skels)
        # error thresholds
        self.t_om = t_om
        self.t_m = t_m
        self.t_s = t_s
        # main results
        self.categories = [None]*self.n_skels
        self.erl_pred = None
        self.erl_gt = None
        self.merge_list = []
        self.split_list = []
        self.corr_list = []
        # GT to pred ID maps
        self.gt2pred = [NodeAssignment(gt_skeletons[i], pred).get_id_maps() for i in range(self.n_skels)]
        
         # go through each skeleton
        for i, sk_outer in enumerate(gt_skeletons):
            # ignore segment 0
            if i==0: 
                self.categories[i] = "omitted"
                continue
            # case: omitted
            omission_flag = (self.gt2pred[i].ids_out[0] == 0
                              and (1.*self.gt2pred[i].counts_out[0])/self.gt2pred[i].size_in > self.t_om)
            if omission_flag:
                self.categories[i] = "omitted"
                continue
            # case: merged
            # check other skeletons in volume for potential merge
            j = i+1
            while j < self.n_skels:
                merged_ids = merge_check(self.gt2pred[i], self.gt2pred[j])
                if len(merged_ids) > 0:
                    self.categories[i] = "merged"
                    self.categories[j] = "merged"
                    for m_id in merged_ids:
                        self.merge_list.append(IDmap(m_id, [i, j]))
                j += 1            
            # case: split or hybrid
            split_ids = split_check(sk_outer, self.gt2pred[i])
            if len(split_ids) > 1:
                if self.categories[i] == "merged": self.categories[i] = "hybrid"
                else: self.categories[i] = "split"
                self.split_list.append(IDmap(i, split_ids))
            # case: correct
            elif self.categories[i] is None:
                self.categories[i] = "correct"
                self.corr_list.append(IDmap(i, split_ids))
                
        if calc_erl==True:
            # compute ERL as weighted mean of individual skeleton ERLs
            # Omitted skeletons are not included in calculation
            # ERL of a skeleton is 0 if merged or hybrid
            # TODO: add method for splits
            tot_length = 0
            pred_length = 0
            gt_length = 0
            for i, sk in enumerate(gt_skeletons):
                if self.categories[i] is "omitted":
                    continue
                this_length = sk.length()
                tot_length += this_length
                gt_length += this_length*this_length
                if self.categories[i] is "correct":
                    pred_length += this_length*this_length
            self.erl_pred = pred_length/tot_length
            self.erl_gt = gt_length/tot_length
        
        print 'Skeleton evaluation time: {}'.format(time.time() - start_time)

    def summary(self, write_path=None):
        n_om = self.categories.count("omitted")
        n_m = self.categories.count("merged")
        n_s = self.categories.count("split")
        n_h = self.categories.count("hybrid")
        n_c = self.categories.count("correct")
        assert n_om+n_m+n_s+n_c+n_h == self.n_skels
        print 'Results:\n%d omissions, %d merges, %d splits, %d hybrid, %d correct'%(n_om, n_m, n_s, n_h, n_c)
        print 'GT ERL: %d, Prediction ERL: %d'%(self.erl_gt, self.erl_pred)
        if write_path is not None:
            cat_counts = {
                "thresholds":{
                    "t_om": self.t_om,
                    "t_m": self.t_m,
                    "t_s": self.t_s
                },
                "results":{
                    "total": self.n_skels,
                    "omitted": n_om,
                    "merged": n_m,
                    "split": n_s,
                    "hybrid": n_h,
                    "correct": n_c,
                    "erl-pred": self.erl_pred,
                    "erl-gt": self.erl_gt
                }
            }
            with open(write_path + '/skeleton-analysis-summary.json', 'w') as outfile:
                json.dump(cat_counts, outfile)
    
    def write_errors(self, write_path):
        # write merges
        f = open(write_path + "/merged-skeletons.ids", "w")
        f.write("%d\n"%(len(self.merge_list))) # no of pairs of merged skeletons
        for merge_item in self.merge_list:
            f.write("%d\n"%(merge_item.id_in)) # ID of pred object
            assert len(merge_item.ids_out) == 2
            f.write("%d, %d\n"%(merge_item.ids_out[0], merge_item.ids_out[1])) # IDs of merged skeletons
        # write splits
        f = open(write_path + "/split-skeletons.ids", "w")
        f.write("%d\n"%(len(self.split_list))) # no of split skeletons
        for split_item in self.split_list:
            f.write("%d\n"%(split_item.id_in)) # ID of split skeleton
            for obj in split_item.ids_out:
                f.write("%d, "%(obj))
        # write correct ids
        f = open(write_path + "/correct-skeletons.ids", "w")
        f.write("%d\n"%(len(self.corr_list))) # no of split skeletons
        for corr_item in self.corr_list:
            f.write("%d, %d\n"%(corr_item.id_in, corr_item.ids_out[0])) # GT id, pred id