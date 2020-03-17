# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:55:48 2020

@author: Adrian
"""

import numpy as np

def kuramoto_step(G, KM=0, dw=0, local=True):
    
    """ 
    Performs one single iteration of the kuramoto model.
    
    Input:
        G -- networkx graph
             required node attributes are:
                 'w' -- characteristic frequency
                 'p' -- oscillator phase
             required edge attributes (if local=True):
                 'K' -- local coupling strength
        KM -- mean-field coupling strength
        dw -- freuency noise amplitude in radians/time
    
    Output:
        dp -- phase change per time, cell array keyed by node index
    
    """
    
    dp = {}
    
    for i in G.nodes:
        
        # get neighbor indices
        
        nbrs = list(G.neighbors(i))
        
        ### phase change due to ...
        
        # ... characteristic frequency
        dp[i] = G.nodes[i]['w']
        
        # ... local coupling
        if nbrs and local:
            dp[i] += 1/G.degree(i) \
                * np.sum([G.edges[i,j]['K']*np.sin(G.nodes[j]['p']-G.nodes[i]['p']) for j in nbrs])
        
        # ... mean-field coupling (excluding self and neighbors)
        if KM:
            non_nbrs = list(set([ID for ID in G.nodes]).difference(set(nbrs+[i])))
            dp[i] += KM/(G.number_of_nodes()-G.degree(i)-1) \
                * np.sum([np.sin(G.nodes[j]['p']-G.nodes[i]['p']) for j in non_nbrs])
        
        # ... frequency noise
        if dw:
            dp[i] += dw*(2*np.random.rand()-1)
            
    return dp
