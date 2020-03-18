# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:55:48 2020

@author: Adrian
"""

import os
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

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

class kuramoto_figure():
    
    """
    Class for plotting and saving of phase data on the underlying graph 
    topologies and on the unit circle.
    
    Input:
        plot_type = {'graph', 'phase', 'both'}
        
    """
    
    def __init__(self, plot_type='graph'):
        
        # plot_type = {'graph', 'phase', 'both'}
        self.plot_type = plot_type
        self.cmap = cm.get_cmap('hsv')
        
        
        if self.plot_type in ['graph']:
            self.figure = plt.figure(figsize=1.5*plt.figaspect(1))
            self.ax = [self.figure.add_subplot()]

        if self.plot_type in ['phase']:
            self.figure = plt.figure(figsize=1.5*plt.figaspect(1))
            self.ax = [self.figure.add_subplot(projection='polar')]
            
        if self.plot_type in ['both']:
            self.figure = plt.figure(figsize=1.5*plt.figaspect(1/2))
            
            gs = gridspec.GridSpec(1, 2)
            self.ax = [self.figure.add_subplot(gs[0,0]), \
                       self.figure.add_subplot(gs[0,1], projection='polar')]

    def plot(self, G):
        
        """
        Update figure with chosen plot_type.
        
        Input:
            G -- networkx graph
                 required node attributes are:
                     'pos' -- 2D node coordinates (x, y)
                     'p' -- phase
                     
        As implemented, the 'hsv' colormap is used.
        
        Node size is scaled by node degree.
        
        """
        
        pos = {n:G.nodes[n]['pos'] for n in G.nodes}

        p = np.asarray([G.nodes[ID]['p'] for ID in G.nodes])
        k = np.asarray([G.degree(ID) for ID in G.nodes])
        
        c = self.cmap(np.mod(p.flatten(),2*np.pi)/(2*np.pi))
        
        if self.plot_type in ['graph', 'both']:
            
            plt.sca(self.ax[0])
            plt.cla()
            
            nx.draw_networkx_edges(G, pos)
            nx.draw_networkx_nodes(G, pos, node_size=10*k, node_color=c)      

            plt.axis('equal')    
            plt.axis('off')

        if self.plot_type in ['phase', 'both']:
            
            plt.sca(self.ax[-1])
            plt.cla()
            
            plt.gca().scatter(np.mod(p.flatten(),2*np.pi), np.ones(G.number_of_nodes()), s=10*k, c=c)
            
            plt.ylim([0,1.5])
            plt.axis('off')
            
        plt.pause(0.001)
        plt.draw()
        
        
    def save(self, Path, Name, dpi=100):
        
        """
        Save current figure.
        
        Input:
            Path -- System path to desired folder. Folder must exist.
            Name -- Output file name including extension.
            dpi -- Output resolution
            
        """
        
        plt.savefig(os.path.join(Path, Name), dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        
def kymograph(t, Gt):
    
    """
    Generate a kymograph of phase data. (Node ID vs time, phase as color)
    
    Input:
        t -- array of time points
        Gt -- dict of graphs, keyed by time points t
        
    Node number does not need to be constant over time.  
    
    """
    
    all_IDs = [list(Gt[ti].nodes()) for ti in Gt]
    all_IDs = [i for sl in all_IDs for i in sl]
    
    nodes = np.unique(all_IDs, axis=0)
    

    nodes = [tuple(ID) if isinstance(ID, np.ndarray) else ID for ID in nodes]
    
    pt = np.zeros((len(nodes), len(Gt)))
    
    for i, t in enumerate(Gt):
        
        G = Gt[t]
        
        for j, ID in enumerate(nodes):

            pt[j,i] = np.mod(G.nodes[ID]['p'], 2*np.pi) if ID in G.nodes else np.nan
        
    
    plt.figure(figsize=plt.figaspect(1/2))
    plt.pcolormesh(np.linspace(0,np.max(t),pt.shape[1]), np.arange(len(nodes)),pt, cmap='hsv')
    plt.xlabel('time', fontsize=14)
    plt.ylabel('node-ID', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14) 
    
    plt.colorbar()