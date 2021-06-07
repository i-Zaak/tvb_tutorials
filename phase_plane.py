from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List
from ipywidgets import interact, FloatSlider, Dropdown
import numpy as np
import matplotlib.pylab as plt


def phase_plane_interactive(model, integrator):
    
    
    NUMBEROFGRIDPOINTS = 42
    
    def plot_phase_plane(**param_kwargs):
        # defaults, to be changed
        svx = param_kwargs.pop('svx') #x-axis: 1st state variable
        svy = param_kwargs.pop('svy') #y-axis: 2nd state variable
        
        
        mode = param_kwargs.pop('mode')

        
        # set model params
        for k, v in param_kwargs.items():
            setattr(model, k, np.r_[v])

        # state vector
        sv_mean = np.array([model.state_variable_range[key].mean() for key in model.state_variables])
        sv_mean = sv_mean.reshape((model.nvar, 1, 1))
        default_sv = sv_mean.repeat(model.number_of_modes, axis=2)
        no_coupling = np.zeros((model.nvar, 1, model.number_of_modes))


        # mesh grid
        xlo = model.state_variable_range[svx][0]
        xhi = model.state_variable_range[svx][1]
        ylo = model.state_variable_range[svy][0]
        yhi = model.state_variable_range[svy][1]

        X = np.mgrid[xlo:xhi:(NUMBEROFGRIDPOINTS*1j)]
        Y = np.mgrid[ylo:yhi:(NUMBEROFGRIDPOINTS*1j)]


        # Calculate the vector field.
        svx_ind = model.state_variables.index(svx)
        svy_ind = model.state_variables.index(svy)


        #Calculate the vector field discretely sampled at a grid of points
        grid_point = default_sv.copy()
        U = np.zeros((NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS,
                              model.number_of_modes))
        V = np.zeros((NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS,
                              model.number_of_modes))
        for ii in range(NUMBEROFGRIDPOINTS):
            grid_point[svy_ind] = Y[ii]
            for jj in range(NUMBEROFGRIDPOINTS):
                #import pdb; pdb.set_trace()
                grid_point[svx_ind] = X[jj]

                d = model.dfun(grid_point, no_coupling)

                for kk in range(model.number_of_modes):
                    U[ii, jj, kk] = d[svx_ind, 0, kk]
                    V[ii, jj, kk] = d[svy_ind, 0, kk]


        # plot
        fig, ax = plt.subplots()
        ax.set(
            xlabel = "State Variable " + svx,
            ylabel = "State Variable " + svy,
            title = model.__class__.__name__ + " mode " + str(mode)
        )
        
        if np.all(U[:, :, mode] + V[:, :, mode]  == 0):
            ax.set(title = model_name + " mode " + mode + ": NO MOTION IN THIS PLANE")
            X, Y = np.meshgrid(X, Y)
            pp_quivers = ax.scatter(X, Y, s=8, marker=".", c="k")
        else:
            pp_quivers = ax.quiver(X, Y,
                                                U[:, :, mode],
                                                V[:, :, mode],
                                                width=0.001, headwidth=8)

        #Plot the nullclines
        nullcline_x = ax.contour(X, Y,
                                              U[:, :, mode],
                                              [0], colors="r")
        nullcline_y = ax.contour(X, Y,
                                              V[:, :, mode],
                                              [0], colors="g")
        plt.show()
        
    # setup widgets 
    param_kwargs = {}
    for param_name in type(model).declarative_attrs:            
            param_def = getattr(type(model), param_name)
            if not isinstance(param_def, NArray) or not param_def.dtype == np.float :
                continue
            param_range = param_def.domain
            if param_range is None:
                continue
            param_value = getattr(model, param_name)[0]
            param_kwargs[param_name] = FloatSlider(
                min=param_range.lo, max=param_range.hi, value=param_value)
    param_kwargs['svx'] = Dropdown(
        #options=[(v,i) for i, v in enumerate(model.state_variables)],
        options = model.state_variables,
        value=model.state_variables[0],
        description='X axis'
    )
    param_kwargs['svy'] = Dropdown(
        #options=[(v,i) for i, v in enumerate(model.state_variables)],
        options = model.state_variables,
        value=model.state_variables[1],
        description='Y axis'
    )
    param_kwargs['mode'] = Dropdown(
        options=list(range(model.number_of_modes)),
        value=0,
        description='Mode'
    )   
    
    w = interact(plot_phase_plane, **param_kwargs)
    return w
