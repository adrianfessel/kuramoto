import networkx as nx
import numpy as np

from tqdm import tqdm

import kuramoto

def import_apollonian_network(Path):
    """
    Import example Physarum polycephalum network.

    Parameters
    ----------
    Path : string
        system path of gpickle file

    Returns
    -------
    G : networkx graph
        networkx graph with positions (x, y) stored as node attributes 'pos'

    References
    ----------
    [1] : http://nbn-resolving.de/urn:nbn:de:gbv:46-00107082-12

    """

    G = nx.read_gpickle(Path)
    G = G.subgraph(max(nx.connected_components(G), key=len))
    pos = {ID:(G.nodes[ID]['x'], G.nodes[ID]['y']) for ID in G.nodes}
        
    for ID in G.nodes:
        G.nodes[ID]['pos'] = pos[ID]
    
    return G
    

def gen_BA_graph(n, m):
    """
    Generate Barabasi-Albert graph using preferential attachment. Positions are 
    generated using the kamada-kawai force method.
    
    Parameters
    ----------
    n : int
        number of nodes
    m : int
        number of edges of each arriving node
             (n = 1 -> tree graph)

    Returns
    -------
    G : networkx graph
        networkx graph with positions (x, y) stored as node attributes 'pos'

    """
    
    G = nx.generators.barabasi_albert_graph(n, m)
    pos = nx.kamada_kawai_layout(G)

    for ID in G.nodes:
        G.nodes[ID]['pos'] = pos[ID]

    return G


def gen_hex_lattice(n, m):
    """
    Generate (non-periodic) hexagonal lattice graph.

    Parameters
    ----------
    n : int
        number of hex fields in x direction
    m : int
        number of hex fields in y direction

    Returns
    -------
    G : networkx graph
        networkx graph with positions (x, y) stored as node attributes 'pos'

    """
    
    G = nx.generators.lattice.hexagonal_lattice_graph(n, m)
    
    return G


def gen_ring_graph(n):
    """
    Generate ring of k=2 nodes. Positions are generated using the kamada-kawai force method.

    Parameters
    ----------
    n : int
        number of nodes

    Returns
    -------
    G : networkx graph
        networkx graph with positions (x, y) stored as node attributes 'pos'

    """
    
    G = nx.Graph()
    G.add_node(0, pos=(0,0))
    for ID in range(1,n):
        G.add_edge(ID-1,ID)
        # G.nodes[ID]['pos'] = (ID, 0)
        
    G.add_edge(ID, 0)
    
    pos = nx.kamada_kawai_layout(G)

    for ID in G.nodes:
        G.nodes[ID]['pos'] = pos[ID]
        
    return G


if __name__ == '__main__':

    # time
    t = 0
    T = 10
    dt = 0.01
    
    # local coupling strength
    KL = 1
    # global coupling strength
    KM = -0.1
    
    # characteristic frequency
    w0 = 2*np.pi
    # frequency noise
    dw = 0*w0
    
    plot = True
    kymograph = True
    
    # generate example graph
    # G = gen_BA_graph(25, 2)
    # G = import_apollonian_network('example_network.pickle')
    # G = gen_hex_lattice(3, 4)
    G = gen_ring_graph(25)
    
    # assign starting phase at random and characteristic frequencies as specified
    for ID in G.nodes:
        G.nodes[ID]['p'] = 2*np.pi*np.random.rand()
        G.nodes[ID]['w'] = w0 + dw*(2*np.random.rand()-1)
        
    # assign local coupling (as implemented: equal weights)
    for ID1, ID2 in G.edges:
        G.edges[ID1, ID2]['K'] = KL
        
    # dict for storing graphs at all time points
    Gt = {}
    
    # iterate model
    pbar = tqdm(total = T/dt)
    while t < T:
        
        dp = kuramoto.kuramoto_step(G, KM, dw)
        
        for ID in G.nodes:
            G.nodes[ID]['p'] += dp[ID]*dt
        
        t += dt
        Gt[t] = G.copy()
        
        pbar.update(1)
        
    pbar.close()
    
    if plot:
        # create figure and play all time steps 
        fig = kuramoto.kuramoto_figure(plot_type='graph')
        
        for i, t in tqdm(enumerate(Gt)):
            
            G = Gt[t]
            
            fig.plot(G)
            # fig.save('C:/Users/Adrian/Desktop/test', str(i) + '.png')
            
    if kymograph:
        # generate kymograph
        kuramoto.kymograph(t, Gt)