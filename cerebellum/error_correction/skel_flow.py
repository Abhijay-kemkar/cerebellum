import numpy as np

from cerebellum.ibex.data_structures.skeleton_points import *

def skel_flow(point, skeleton):
    """
    returns best non-horizontal flow vector at point based on nearest skeleton edge
    
    vector is normalized such that z-component = 1
    """
    
    def dist_pt2ln(pt, ln):
        """
        finds euclidean distance between point and line
        Args:
            pt (3x, array)
            ln (2x3 array)
        """
        p1 = ln[0,:]
        p2 = ln[1,:]
        return np.linalg.norm(np.cross(p2-p1, p1-pt))/np.linalg.norm(p2-p1)
    
    try:
        assert type(point) is list and len(point)==3
    except:
        print point
    nodes = skeleton.get_nodes()
    dists = np.array([np.linalg.norm(np.array(node-point)) for node in nodes])
    nodes = [nodes[i] for i in np.argsort(dists)]
    #print nodes
    edges = skeleton.get_edges()
    edge_vecs = [edge[0]-edge[1] for edge in edges]
    edge_vecs = [np.divide(edge_vec, float(edge_vec[0])) for edge_vec in edge_vecs] # normalize such that z-comp is 1
    # iterate over nodes till you find a non-horizontal edge at a node closest to point
    found_flow = False
    check_edges = []
    while not found_flow:
        try:
            check_node = nodes.pop(0) # closest node to input point
            #print check_node
        except:
            print "Error! Could not find non-horizontal flow vector at this point"
        for i, (edge, edge_vec) in enumerate(zip(edges, edge_vecs)):
            allow_edge = ((np.all(check_node==edge[0]) or np.all(check_node==edge[1])) 
                             and not np.any(np.isnan(edge_vec)) and not np.any(np.isinf(edge_vec)))
            if allow_edge:
                check_edges.append(i)
        found_flow = (len(check_edges)>0)

    #print edges[check_edges]
    #print check_node
    # find edges for which the point is z-midway
    midway_edges = []
    for i in check_edges:
        z_lower = min(edges[i][0,0], edges[i][1,0])
        z_higher = max(edges[i][0,0], edges[i][1,0])
        if z_lower > point[0] or z_higher < point[0]:
            midway_edges.append(i)
    if len(midway_edges)>0: 
        check_edges = midway_edges
    # find edge closest to point
    check_dists = [dist_pt2ln(np.array(point), edges[i]) for i in check_edges]
    flow_edge_id = check_edges[np.argmin(check_dists)]
    #print edges[flow_edge_id]
    flow_vec = edge_vecs[flow_edge_id]
    return flow_vec