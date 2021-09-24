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

def _good_history_shape(conn, dt, model):
    idelays = np.rint(conn.delays / dt).astype(np.int32)
    horizon = idelays.max() + 1    
    nvar = model.nvar
    nnodes = conn.number_of_regions
    nmodes = model.number_of_modes

    return (horizon, nvar, nnodes, nmodes)

def _generate_initial_conditions(svar_range, shape):
    nt, nvar, nnode, nmode = shape
    ic = np.empty(shape)
    svr = svar_range
    sv = list(svar_range.keys())
    block = nt, nnode, nmode
    for i, (lo, hi) in enumerate([svr[sv[i]] for i in range(nvar)]):
        ic[:, i] = np.random.uniform(low=lo, high=hi, size=block)
    return ic

def initial_conditions(model, conn, dt, svar_range):
    shape = _good_history_shape(conn, dt, model)
    svar_range_full = model.state_variable_range.copy()
    for key, val in svar_range.items():
        svar_range_full[key] = val
    return _generate_initial_conditions(svar_range_full, shape)