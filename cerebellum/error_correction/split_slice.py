from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import itertools
import numpy as np
import matplotlib.pyplot as plt

from skel_flow import skel_flow

def split_slice(fill_slice, target_ccs, seed_method="skel-flow",
                footprint=None, 
                pivot_slice=None, pivot_id=None, fill_dir=None, bbox_offset=None, 
                seg_res = None, skeleton=None,
                prop_from="dist-xform", tweak_boundary_seeds=True, min_seed_connectivity=1,
                save_path=None, plot_seeds=False):

    """
    Function to split object in a slice given a pivot slice with more connected components
    TO DO: accept params in dict instead of args list
    """

    def snap(pt, image):
        """
        Checks if pt lies outside image extents, snaps to nearest pt inside image grid
        Args:
            pt (Nx, array)
            image (N dim array)
        """
        new_pt = list(pt)
        for i in range(len(image.shape)):
            if pt[i]<0:
                new_pt[i] = 0
            elif pt[i]>=image.shape[i]:
                new_pt[i] = image.shape[i]-1
        return tuple(new_pt)
    
    def on_boundary(pt, image, connectivity=2):
        """
        Checks if pt is on a boundary in a binary image
        Args:
            pt (2x, array)
            image (2 dim array)
            connectivity (int)
        """
        assert len(pt)==2
        assert len(np.unique(image))==2
        assert connectivity==1 or connectivity==2
        # generate neighbors
        if connectivity==2:
            neighbs = [[(pt[0]+d0, pt[1]+d1) for d0 in [-1,0,1]] for d1 in [-1,0,1]]
            neighbs = list(itertools.chain.from_iterable(neighbs))
            neighbs.remove(pt)
        elif connectivity==1:
            neighbs = [(pt[0]+1,pt[1]), (pt[0]-1,pt[1]), (pt[0],pt[1]-1), (pt[0],pt[1]+1)]
        neighbs = [snap(nb, image) for nb in neighbs]
        if any([image[nb]==0 for nb in neighbs]):
            return True
        else:
            return False
        
    def push_inwards(pt, image, target_conn=2):
        """
        Pushes a pt on boundary of image inwards, tries to return best connected neighbor
        Args:
            pt (2x, array)
            image (2 dim array)
        """
        assert target_conn>0
        try:
            assert on_boundary(pt, image, connectivity=target_conn)
        except:
            print "Point already inwards, returning as is"
            return pt
        neighbs = [[(pt[0]+d0, pt[1]+d1) for d0 in [-1,0,1]] for d1 in [-1,0,1]]
        neighbs = list(itertools.chain.from_iterable(neighbs))
        neighbs.remove(pt)
        neighbs = [snap(nb, image) for nb in neighbs]
        one_neighbs = [nb for nb in neighbs if not on_boundary(nb, image, connectivity=1)]
        two_neighbs = [nb for nb in neighbs if not on_boundary(nb, image, connectivity=2)]
        if len(two_neighbs)>0 and target_conn<=2:
            return two_neighbs[0] # arbitraily chosen
        elif len(two_neighbs)==0 and len(one_neighbs)>0 and target_conn==1:
            return one_neighbs[0] # arbitraily chosen
        else:
            print "Warning: Failed to find better connected neighbor. Returning point as is"
            return pt
    
    assert seed_method=="dist-xform" or seed_method=="skel-flow"
    
    distance = ndi.distance_transform_edt(fill_slice) # could also use mean affinity map here
    
    # generate seeds for watershed
    # Method 1: local maxima of distance xform of fill slice
    if seed_method=="dist-xform":
        assert type(footprint) is np.ndarray
        seed_locs = peak_local_max(distance, indices=False, num_peaks=2, footprint=footprint) # TODO: CCs>2
        seeds = ndi.label(seed_locs)[0]
        
    # Method 2: use flow vector from skeleton
    elif seed_method=="skel-flow":
        assert prop_from=="centroids" or prop_from=="dist-xform"
        assert fill_dir==-1 or fill_dir==1
        assert pivot_slice.shape == fill_slice.shape
        assert type(pivot_id) is int
        assert bbox_offset is not None and len(bbox_offset)==2
        assert skeleton is not None
        if tweak_boundary_seeds: assert min_seed_connectivity>0
        pivot_cc_ids = np.unique(pivot_slice).tolist()
        pivot_cc_ids.remove(0)
        assert target_ccs == len(pivot_cc_ids)
        
        # Choose points in pivot slice to propagate
        pivot_regions = [np.zeros_like(pivot_slice) for _ in range(target_ccs)]
        for i, pr in enumerate(pivot_regions):
            pr[pivot_slice==pivot_cc_ids[i]] = 1
        # Method 1: centroids
        if prop_from=="centroids":
            pivot_centroids = [np.mean(np.argwhere(pr!=0), axis=0) for pr in pivot_regions]
            prop_pts = pivot_centroids
        # Method 2: local maxima of distance transform of each CC
        elif prop_from=="dist-xform":
            pivot_distances = [ndi.distance_transform_edt(pr) for pr in pivot_regions]
            pivot_dx_centers = [peak_local_max(pd, indices=True, num_peaks=1)[0] for pd in pivot_distances]
            prop_pts = pivot_dx_centers
        print "Pivot points for propagation:", prop_pts
        
        # Estimate flow
        assert skeleton.resolution[0]==seg_res[0]
        dsmpl = (seg_res[1]/skeleton.resolution[1], seg_res[2]/skeleton.resolution[2])
        flow_sources = [[pivot_id, (p[0]+bbox_offset[0])/dsmpl[0], (p[1]+bbox_offset[1])/dsmpl[1]] 
                        for p in prop_pts] # repackage for skel_flow()
        #print "Flow sources:", flow_sources
        flow_vectors = [skel_flow(point, skeleton) for point in flow_sources]
        print "Flow vectors:", [f.tolist() for f in flow_vectors]
        
        # Generate seeds
        seed_locs = [(int(p[0]+fill_dir*f[1]), 
                    int(p[1]+fill_dir*f[2])) for (p, f) in zip(prop_pts, flow_vectors)]
        print "Seed locs:", seed_locs
        seeds = np.zeros_like(fill_slice)
        seeds_valid = [False for _ in range(target_ccs)]
        for i, seed_loc in enumerate(seed_locs):
            # TODO: Re-enable tweak_boundary options after fixing on_boundary and push_inwards to operate on images with >1 CC
            #  if on_boundary(seed_loc, fill_slice, connectivity=min_seed_connectivity):
            #    if tweak_boundary_seeds:
            #        print "Warning: Skel-flow seed on boundary of fill slice. Will attempt to push inwards"
            #        seed_loc = push_inwards(seed_loc, fill_slice, target_conn=min_seed_connectivity)
            #    else:
            #        print "Warning: Skel-flow seed on boundary of fill slice. Ignoring and proceeding with watershed"
            seeds_valid[i] = fill_slice[seed_loc]>0
            if seeds_valid[i]:
                seeds[seed_loc] = i+1
        final_seed_count = sum(seeds_valid)

    if final_seed_count != target_ccs:
        print "Warning: Skel-flow seed(s) generated in empty region of fill slice. "+ \
               "Fill slice will have %d CCs instead of requested %d CCs"%(final_seed_count, target_ccs)

    # run watershed
    filled_slice = watershed(-distance, seeds, mask=fill_slice)

    # plot original slice with seeds and watershedded slice for debugging
    if plot_seeds:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,15))
        ax1.imshow(fill_slice, alpha=0.5)
        ax1.imshow(seeds, alpha=0.5)
        ax2.imshow(distance)
        ax3.imshow(filled_slice)
        plt.show()
    # plot watershedded slice only
    if save_path is not None:
        fig, ax = plt.subplots(1,1,figsize=(15,15))
        ax.imshow(filled_slice)
        plt.savefig(save_path)
        plt.close(fig)
    
    return filled_slice, final_seed_count