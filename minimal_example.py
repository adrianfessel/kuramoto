# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:55:25 2020

@author: Adrian
"""

import networkx as nx
import numpy as np

from tqdm import tqdm

from kuramoto import kuramoto_step, kuramoto_figure

# parameters

t = 0
T = 10
dt = 0.025

KL = 1
KM = 0

w0 = 2*np.pi
dw = 0*w0

# generate example graph

n = 100
m = 1

G = nx.generators.barabasi_albert_graph(n, m)
pos = nx.kamada_kawai_layout(G)

for ID in G.nodes:
    G.nodes[ID]['pos'] = pos[ID]
    G.nodes[ID]['p'] = 2*np.pi*np.random.rand()
    G.nodes[ID]['w'] = w0
    
for ID1, ID2 in G.edges:
    G.edges[ID1, ID2]['K'] = KL
    
Gt = {}

pbar = tqdm(total = T/dt)

while t < T:
    
    dp = kuramoto_step(G, KM, dw)
    
    for ID in G.nodes:
        G.nodes[ID]['p'] += dp[ID]*dt
    
    t += dt
    Gt[t] = G.copy()
    
    pbar.update(1)
    
pbar.close()

fig = kuramoto_figure(plot_type='graph')

for i, t in tqdm(enumerate(Gt)):
    
    G = Gt[t]
    
    fig.plot(G)
    # fig.save('C:/Users/Adrian/Desktop/test', str(i) + '.png')