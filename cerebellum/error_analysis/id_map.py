class IDmap(object):
    """Map from an ID of one segmentation/skeletonization to IDs in another"""
    def __init__(self, id_in, ids_out, size_in=0, counts_out=None):
        """
        Attributes:
            id_in (int): ID of input object
            size_in (int): Size of input object in # voxels or # nodes
            ids_out (list of ints): IDs of intersecting objects in output volume
            counts_out (list of ints): Size of intersection in # voxels or # nodes
        """
        if counts_out is None:
            counts_out = [0]*len(ids_out)
        assert len(ids_out) == len(counts_out)
        assert type(id_in) == type(size_in) == int
        self.id_in = id_in
        self.size_in = size_in
        self.ids_out = ids_out
        self.counts_out = counts_out