import numpy as np
import emcee
import multiprocessing
import datetime
import pickle

try:
    # for Jupyter
    from tqdm.notebook import tqdm
except ImportError:
    # for terminal
    from tqdm import tqdm

SOLUTION_TYPES = [0,5,7,9,12]
### --- ###
def calculate_orbit_parameter(m, q, w):
    ''' This is lambda
    '''
    return q*w*m**(1/3)*(1 + q)**(-2/3)

### --- ###
def q_from_l(l, m, w):
    '''
        sort of a nightmare to disentangle the nonlinear q dependence
        in lambda, this function solves it numerically if that's ever
        needed
    '''
    z = m * (w / l)**3
    # Coefficients of z q^3 - q^2 - 2q - 1 = 0
    coeff = [z, -1.0, -2.0, -1.0]
    roots = np.roots(coeff)
    # real roots only
    real_roots = roots[np.isreal(roots)].real
    # choose the physically valid one: q > 0
    valid = real_roots[real_roots > 0]
    if len(valid) == 0:
        return -1.0
    # Usually only one positive root exists
    return valid[0]

### --- ###
def q_from_l_vectorized(l_array, m, w):
    '''
        vectorised version of q_from_l()
    '''
    z = m * (w / l_array)**3

    # coefficients for all cubics
    coeffs = np.column_stack([z, -np.ones_like(z), -2*np.ones_like(z), -1*np.ones_like(z)])
    roots = np.array([np.roots(c) for c in coeffs])  # shape (N, 3)

    # real roots mask
    real_roots = roots.real * np.isreal(roots)  # imaginary parts removed

    # positive roots mask
    positive_mask = real_roots > 0

    # pick the first positive root (there should be exactly one)
    q_vals = np.where(positive_mask.any(axis=1),
                      real_roots[np.arange(len(real_roots)), positive_mask.argmax(axis=1)],
                      -1.0)
    return q_vals

### --- ###
def rescale_period(cube, periods, period_boundaries):
    bin_indices = np.searchsorted(period_boundaries, periods, side="right")

    # Efficient accumulation
    reshaped_cube = np.zeros((len(period_boundaries) + 1, cube.shape[1]), dtype=cube.dtype)
    np.add.at(reshaped_cube, bin_indices, cube)

    return reshaped_cube

### --- ###
def rescale_lambda(target_object, cube, lambdas, m2_boundaries, 
                   m2lim=(0.017, 0.2), qlim=(0.05, 0.5)):

    # Precompute masses and q
    mass = target_object["mass"]
    mass_ratios = q_from_l_vectorized(lambdas, mass, target_object["parallax"])
    companion_masses = mass * mass_ratios

    # Mask of acceptable lambda points
    valid_mask = (
        (companion_masses > m2lim[0]) &
        (companion_masses < m2lim[1]) &
        (mass_ratios > qlim[0]) &
        (mass_ratios < qlim[1])
    )

    # Get valid indices
    valid_idx = np.where(valid_mask)[0]

    # Bin companion masses
    col_bins = np.searchsorted(m2_boundaries, companion_masses[valid_idx], side="right")

    # Output array
    q_space_cube = np.zeros((cube.shape[0], len(m2_boundaries) + 1), dtype=cube.dtype)

    # Add contributions column-wise
    np.add.at(q_space_cube, (slice(None), col_bins), cube[:, valid_idx])

    return q_space_cube

### --- ###
def compute_grid(target_object, sc_cubes, period_boundaries, m2_boundaries):
    plx_index = np.argmin(abs(sc_cubes["meta"]["parallaxes"] - target_object["parallax"]))
    reference_plx = sc_cubes["meta"]["parallaxes"][plx_index]
    
    # choose the right precomputed grid, with the right solution type
    working_cube = sc_cubes["data"][reference_plx][:,:,target_object["soltype_index"]]
    
    # scale it down to the right period binning
    period_scaled_cube = rescale_period(working_cube, np.log10(sc_cubes["meta"]["periods"]), period_boundaries)
    
    # and scale it to the working companion mass
    fully_rescaled_cube = rescale_lambda(target_object, period_scaled_cube, sc_cubes["meta"]["lambdas"][plx_index], m2_boundaries)
    
    return fully_rescaled_cube/fully_rescaled_cube.sum()
        
### --- ###
def compute_grids(objects, sc_cubes, period_boundaries, m2_boundaries):
    grids = []
    
    # for every object, first map to the scaled down period resolution
    # then, map it into m2-space, trimming the irrelevant m2s
    for target_object in tqdm(objects):
        # identify the right 'cube', with the closest reference parallax, for the job
        plx_index = np.argmin(abs(sc_cubes["meta"]["parallaxes"] - target_object["parallax"]))
        reference_plx = sc_cubes["meta"]["parallaxes"][plx_index]
        
        # choose the right precomputed grid, with the right solution type
        working_cube = sc_cubes["data"][reference_plx][:,:,target_object["soltype_index"]]
        
        # scale it down to the right period binning
        period_scaled_cube = rescale_period(working_cube, np.log10(sc_cubes["meta"]["periods"]), period_boundaries)
        
        # and scale it to the working companion mass
        fully_rescaled_cube = rescale_lambda(target_object, period_scaled_cube, sc_cubes["meta"]["lambdas"][plx_index], m2_boundaries)
        fully_rescaled_cube = fully_rescaled_cube/fully_rescaled_cube.sum()
        grids.append(fully_rescaled_cube.ravel())
    
    return grids   

### --- ###
def calculate_log_likelihood(mcmc_params, soltypes, grids, cutoff=np.exp(-18)): 
    ## make sure we're within the prior
    fb = mcmc_params[0]
    grid_params = mcmc_params[1:]
    
    # make sure we're within fb prior
    if (fb < 0) | (fb > 1):
        return -np.inf
    
    # convert real valued parameters to logical probability distribution
    # using softmax so that we don't have to deal with multinomials
    grid_params = mcmc_params[1:]
    exp_x = np.exp(grid_params - grid_params.max())
    grid_probs = exp_x / exp_x.sum()
    
    # compute individual solution chance
    dot_values = fb * np.dot(grids, grid_probs)

    # For solution_type == 0, add single star component (1 - fb)
    dot_values[soltypes == 0] += (1 - fb)

    # Apply cutoff and sum log-likelihoods
    return np.sum(np.log(np.maximum(dot_values, cutoff)))
    
### --- ###
class popsampler():
    def __init__(self, sc_cubes, catalogue, p_boundaries, m_boundaries):
        self.sc_cubes = sc_cubes
        self.catalogue = catalogue
        
        self.sampler = None
        self.p_boundaries = p_boundaries
        self.m_boundaries = m_boundaries
        self.grid_shape = np.array([len(p_boundaries)+1, len(m_boundaries)+1])
        
    ### --- ###
    def constrain_parameters(self, step_count=30, nwalkers=50, initialisation_weight=1e-2, cutoff=np.exp(-18)):
        temp_kwargs = dict()
        temp_kwargs["cutoff"] = cutoff
        
        print("Reducing catalogue...")
        working_catalogue = []
        soltypes = np.zeros(len(self.catalogue), dtype=np.int8)
        for i, target_object in enumerate(self.catalogue):
            # save just parallax and mass for grid conversion
            reduced_object = {
                "parallax": target_object["parallax"],
                "mass": target_object["mass"],
                "soltype_index": SOLUTION_TYPES.index(target_object["solution_type"])
            }
            working_catalogue.append(reduced_object)
            
            # save just solution type for use at inference
            soltypes[i] = target_object["solution_type"] 
        working_catalogue = np.array(working_catalogue)

        # precompute the q-L mappings for all the objects
        print("Computing grids...")
        grids = np.array(compute_grids(working_catalogue, self.sc_cubes, self.p_boundaries, self.m_boundaries))
        
        # run mcmc
        print("Running markov chains...")
        args = (soltypes, grids)
        self.sampler = None
        ndim = 1+self.grid_shape.prod()
        initial_params = np.zeros((nwalkers, ndim))
        # prior on binary fraction is flat
        initial_params[:,0] = np.random.rand(nwalkers)
        # prior on grid is sampled around zero
        initial_params[:,1:] = initialisation_weight * np.random.randn(nwalkers, ndim-1)
        pool = multiprocessing.Pool()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_log_likelihood, 
                                    args=args, kwargs=temp_kwargs, pool=pool)
        sampler.run_mcmc(initial_params, step_count, progress=True, skip_initial_state_check=True)
        print("Complete!")
        self.sampler = sampler
    
    ### --- ###
    def save_results(self, name, save_dir=None, note=None):          
        outdata = dict()
        outdata["metaparams"] = dict()
        outdata["metaparams"]["name"] = name
        outdata["metaparams"]["notes"] = note
        outdata["metaparams"]["timestamp"] = datetime.datetime.now()
        
        outdata["sampler"] = self.sampler
        
        if save_dir is None:
            save_dir = "."
            
        outfile = open("{}/{}.pkl".format(save_dir, name), "wb")
        pickle.dump(outdata, outfile)
        outfile.close()