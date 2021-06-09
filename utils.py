from tvb.simulator.lab import *
import numpy as np
import matplotlib.pylab as plt


def plot_connectivity(conn):
    # Visualization
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle('TVB SC', fontsize=20)

    # Weights
    plt.subplot(121)
    plt.imshow(conn.weights, interpolation='nearest', aspect='equal', cmap='jet')
    plt.xticks(range(0, conn.number_of_regions), conn.region_labels, fontsize=7, rotation=90)
    plt.yticks(range(0, conn.number_of_regions), conn.region_labels, fontsize=7)
    cb = plt.colorbar(shrink=0.2)
    cb.set_label('Weights', fontsize=14)

    # Tracts lengths
    plt.subplot(122)
    plt.imshow(conn.tract_lengths, interpolation='nearest', aspect='equal', cmap='jet')
    plt.xticks(range(0, conn.number_of_regions), conn.region_labels, fontsize=7, rotation=90)
    plt.yticks(range(0, conn.number_of_regions), conn.region_labels, fontsize=7)
    cb = plt.colorbar(shrink=0.2)
    cb.set_label('Tract lenghts', fontsize=14)

    fig.tight_layout()

    plt.show()

def generate_initial_conditions_array(conn, dt, model):
    idelays = np.rint(conn.delays / dt).astype(numpy.int32)
    horizon = idelays.max() + 1    
    nvar = model.nvar
    nnodes = conn.number_of_nodes
    nmodes = model.number_of_modes

    return np.zeros( (horizon, nvar, nnodes, nmodes) )    