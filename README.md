# kuramoto
	Simple implementation of the Kuramoto model [1] for synchronization of phase oscillators 
	with local and mean-field coupling. Provides basic functions for graph generation and plotting.
	
	The Kuramoto model describes the interaction of oscillators by adjusting the progression of their 
	oscillation phase based on the phase difference to all oscillators it is coupled to. This coupling can 
	be global, i.e., each oscillator is influenced by all others, or local, meaning that each oscillator
	is coupled only to its neighbors. In the latter case, the topology of the coupling can be described as
	a graph. Phenomena that can be observed depend on the coupling topology and strength, and as well the
	presence of external noise. An illustration is shown below.
	
![alt text](https://github.com/adrianfessel/kuramoto/blob/master/expl_figure.png?raw=true)
	
# Contents
	kuramoto.py : Function for iterating the Kuramoto model; functions for plotting
	main.py : Script for executing the model; functions for graph generation

# References
	[1] : https://doi.org/10.1016/j.physrep.2015.10.008
	[2] : https://doi.org/10.1098/rstb.2019.0757 (source of the figure)
