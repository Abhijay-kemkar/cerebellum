from skimage.morphology import label
import matplotlib.pyplot as plt
import numpy as np

from split_slice import split_slice

class SplitObject(object):
    """
    Methods to split a single object in a segmentation
    """
    def __init__(self, shape_mask, seg_res, bbox, skeleton):
        self.shape_mask = shape_mask
        self.seg_res = seg_res
        self.bbox = bbox # bbox of object in overall segmentation
        self.skeleton = skeleton

        self.n_slices =shape_mask.shape[0]
        self.skel_res = skeleton.resolution
        self.cc_mask = None
        self.n_cc = np.zeros(self.n_slices, dtype=int)
        self.fill_nbds = None
        self.corr_slices = None
        self.corr_cc_mask = None
        self.corr_ncc = None

    def label_slices(self, connectivity=2, save_plots_path=None):
        """Labels connected components in each slice"""
        self.cc_mask = np.zeros_like(self.shape_mask, dtype=int)
        plt.ioff()
        for slice_id in range(self.n_slices):
            self.cc_mask[slice_id,:,:], self.n_cc[slice_id] = label(self.shape_mask[slice_id,:,:], 
                                                                    connectivity=connectivity, return_num=True)
            if save_plots_path is not None:
                fig, ax = plt.subplots(1,1,figsize=(15,15))
                ax.imshow(self.cc_mask[slice_id,:,:])
                plt.savefig(save_plots_path+'slice%d.png'%(slice_id))
                plt.close(fig)

    def plot_ncc(self):
        plt.scatter(np.arange(self.n_slices), self.n_cc, c="g", alpha=0.5)
        plt.xlabel("slice id")
        plt.ylabel("CC #")
        plt.show()

    def find_fill_nbds(self):
        """Finds fill neighborhoods after connected components per slice are found"""
        assert all(self.n_cc>0)
        assert self.fill_nbds is None
        # generate pivotal slices and fill directions
        pivot_slices = [] # frontier slices with max # CC
        fill_dirs = [] # direction in which CCs must be propagated from corresponding pivotal slices
        self.fill_nbds = []
        max_ccs = max(self.n_cc) # only slices with max_ccs # CCs are deemed correct initially
        for i in range(self.n_slices):
            if self.n_cc[i]==max_ccs:
                if i>0 and self.n_cc[i-1]<max_ccs:
                    pivot_slices.append(i)
                    fill_dirs.append(-1)
                if i<self.n_slices-1 and self.n_cc[i+1]<max_ccs:
                    pivot_slices.append(i)
                    fill_dirs.append(1)
                if len(pivot_slices)>1 and pivot_slices[-1]==pivot_slices[-2]==i:
                    del pivot_slices[-1]
                    del fill_dirs[-1]
                    fill_dirs[-1]=0
        # print zip(pivot_slices, fill_dirs)

        # generate fill neighborhoods
        # each neighborhood is of the form [pivotal slice, terminal slice]
        for i in range(len(pivot_slices)):
            if fill_dirs[i]==-1 or fill_dirs[i]==0:
                if i==0:
                    self.fill_nbds.append([pivot_slices[i], 0])
                else:
                    self.fill_nbds.append([pivot_slices[i], (pivot_slices[i-1]+pivot_slices[i])/2+1])
            if fill_dirs[i]==1 or fill_dirs[i]==0:
                if i==len(pivot_slices)-1:
                    self.fill_nbds.append([pivot_slices[i], self.n_slices-1])
                else:
                    self.fill_nbds.append([pivot_slices[i], (pivot_slices[i+1]+pivot_slices[i])/2])

    def propagate_slice(self, pivot_id, fill_id, use_updated= False, save_path=None, plot_seeds=False):
        """Propagates CCs in given pivot slice into requested fill slice"""
        fill_slice = self.cc_mask[fill_id,:,:].copy()
        fill_dir = int(np.sign(fill_id-pivot_id))
        if not use_updated: # work with uncorrected mask provided intiially
            pivot_slice = self.cc_mask[pivot_id,:,:].copy()
            target_ccs = self.n_cc[pivot_id]
        else: # use correct CC mask as it is being updated for pivot slice
            pivot_slice = self.corr_cc_mask[pivot_id,:,:].copy()
            target_ccs = self.corr_ncc[pivot_id]
        filled_slice, filled_ccs = split_slice(fill_slice, target_ccs=target_ccs,
                                                seed_method="skel-flow", 
                                                pivot_slice=pivot_slice, pivot_id=pivot_id,
                                                fill_dir=fill_dir, bbox_offset=(self.bbox[1], self.bbox[2]), 
                                                seg_res=self.seg_res, skeleton=self.skeleton,
                                                prop_from="dist-xform", tweak_boundary_seeds=False,
                                                save_path=save_path, plot_seeds=plot_seeds)
        return filled_slice, filled_ccs

    def recalc_fill_nbds(self):
        """Recompute fill neighborhoods after some slices are corrected"""
        current_nbds = self.fill_nbds[:]
        # generate new pivotal slices and fill directions
        pivot_slices = [] # frontier slices with max # CC
        fill_dirs = [] # direction in which CCs must be propagated from corresponding pivotal slices
        for i in range(self.n_slices):
            if i in self.corr_slices:
                if i>0 and i-1 not in self.corr_slices:
                    pivot_slices.append(i)
                    fill_dirs.append(-1)
                if i<self.n_slices-1 and i+1 not in self.corr_slices:
                    pivot_slices.append(i)
                    fill_dirs.append(1)
                if len(pivot_slices)>1 and pivot_slices[-1]==pivot_slices[-2]==i:
                    del pivot_slices[-1]
                    del fill_dirs[-1]
                    fill_dirs[-1]=0
        # print zip(pivot_slices, fill_dirs)

        # generate fill neighborhoods
        # each neighborhood is of the form [pivotal slice, terminal slice]
        new_nbds = []
        for i in range(len(pivot_slices)):
            if fill_dirs[i]==-1 or fill_dirs[i]==0:
                if i==0:
                    new_nbds.append([pivot_slices[i], 0])
                else:
                    new_nbds.append([pivot_slices[i], (pivot_slices[i-1]+pivot_slices[i])/2+1])
            if fill_dirs[i]==1 or fill_dirs[i]==0:
                if i==len(pivot_slices)-1:
                    new_nbds.append([pivot_slices[i], self.n_slices-1])
                else:
                    new_nbds.append([pivot_slices[i], (pivot_slices[i+1]+pivot_slices[i])/2])

        return new_nbds

    def fill_object(self, save_path=None):
        """
        Fills entire object by attempting to conserve # CC's through slices

        Splits all slices with false merges
        """
        # initialize list of correct slice IDs
        self.corr_slices = range(self.n_slices)
        for nbd in self.fill_nbds:
            wrong_ids = range(nbd[1], nbd[0], int(np.sign(nbd[0]-nbd[1])))
            for i in wrong_ids:
                self.corr_slices.remove(i)
        # initialize slices with correct CCs
        self.corr_ncc = [0 for _ in range(self.n_slices)]
        self.corr_cc_mask = np.zeros_like(self.cc_mask)
        for i in self.corr_slices:
            self.corr_ncc[i] = self.n_cc[i]
            self.corr_cc_mask[i,:,:] = self.cc_mask[i,:,:]
        # initialize fill neighborhoods
        current_nbds = self.fill_nbds[:]

        # till all slices are correct
        n_rounds = 0
        while len(self.corr_slices) < self.n_slices:
            print "\n\n\n%d slices left to fill"%(self.n_slices-len(self.corr_slices))
            # work on each fill neighborhood
            for nbd in current_nbds:
                pivot_id = nbd[0]
                terminal_id = nbd[1]
                print "\n\nInspecting neighborhood:", nbd
                fill_dir = int(np.sign(terminal_id-pivot_id))
                terminate_fill = False
                # till entire neighborhood is filled or at least one CC could not be tracked into fill slice
                while pivot_id!=terminal_id:
                    fill_id = pivot_id + fill_dir
                    fill_ccs = self.n_cc[fill_id]
                    target_ccs = self.corr_ncc[pivot_id]
                    # case: CC # is already conserved
                    if fill_ccs == target_ccs:
                        self.corr_cc_mask[fill_id,:,:] = self.cc_mask[fill_id,:,:]
                        self.corr_ncc[fill_id] = fill_ccs
                        self.corr_slices.append(fill_id)
                        pivot_id += fill_dir
                    # case: CC # decreases
                    elif fill_ccs < target_ccs:
                        print "\nTracking %d and splitting %d into %d CCs"%(pivot_id, fill_id, target_ccs)
                        self.corr_cc_mask[fill_id,:,:], self.corr_ncc[fill_id] = self.propagate_slice(pivot_id, fill_id,
                                                                                                      use_updated=True,
                                                                                                      save_path=save_path+'slice%d.png'%(fill_id))
                        self.corr_slices.append(fill_id)
                        terminate_fill = (self.corr_ncc[fill_id] < self.corr_ncc[pivot_id]) # CC break detected
                        if terminate_fill:
                            print "Filling terminated at slice %d"%(fill_id)
                            break
                        else:
                            pivot_id += fill_dir # use newly filled slice as source for next fill
                    # case: CC # increases. not allowed by definition of pivot slice
                    else:
                        raise Exception("detected pivot slice %d with lesser CCs than fill slice %d"%(pivot_id, fill_id))

            # update fill neighborhoods
            print "\n\nFinished one round of correction. Recalculating fill neighborhoods"
            current_nbds = self.recalc_fill_nbds()
            n_rounds += 1

        print "\n\n\nObject filling complete after %d rounds of filling"%(n_rounds)