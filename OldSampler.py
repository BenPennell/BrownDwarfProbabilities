import numpy as np
import emcee
import multiprocessing
import datetime
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import corner

try:
    # for Jupyter
    from tqdm.notebook import tqdm
except ImportError:
    # for terminal
    from tqdm import tqdm

# to save on memory, make a global variable for the grids
_global_grids = None

def set_global_grids(grids):
    global _global_grids
    _global_grids = grids

SOLUTION_TYPES = [0,5,7,9,12]
### --- ###
def calculate_orbit_parameter(m, q, w):
    ''' This is lambda
    '''
    return q*w*m**(1/3)*(1 + q)**(-2/3)

### --- ###
def convert_binarity(fb, a):
    return a / (a + 1/fb - 1)

### --- ###
def scale_resolution(arr, scale=2, axis=0, even=False):
    '''
        upscales grid resolution horizontally by splitting grid values evenly into multiple cells
    '''
    # Create a new shape with double the size along the specified axis
    new_shape = list(arr.shape)
    new_shape[axis] *= scale

    # Expand the array along a new axis after the target one
    expanded = np.expand_dims(arr, axis + 1)  # shape becomes (..., 1, ...)
    
    # Repeat the values along the new axis (splitting them evenly)
    repeated = np.repeat(expanded, scale, axis=axis + 1)
    if even:
        repeated = repeated / scale

    # Reshape back by merging the expanded axis with the original one
    transposed = np.reshape(repeated, new_shape)

    return transposed

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
def rescale_period(cube, periods, period_boundaries, plim=(1,8)):
    bin_indices = np.searchsorted(period_boundaries, periods, side="right")

    # Mask of acceptable lambda points
    # valid_mask = (
    #     (periods > plim[0]) &
    #     (periods < plim[1])
    # )
    
    # sum up the corresponding rows
    reshaped_cube = np.zeros((len(period_boundaries) + 1, cube.shape[1]), dtype=cube.dtype)
    np.add.at(reshaped_cube, bin_indices, cube)
    
    # we want the mean, so work out counts to divide by
    counts = np.bincount(bin_indices, minlength=reshaped_cube.shape[0])
    mask_empty = counts > 0
    reshaped_cube[mask_empty] /= counts[mask_empty, None]
    
    return reshaped_cube

### --- ###
def rescale_lambda(target_object, cube, lambdas, m2_boundaries, 
                   m2lim=(0.017, 0.2), qlim=(0.05, 0.5), save_cols=True):

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
    
    # we want the mean, so work out counts to divide by
    counts = np.bincount(col_bins, minlength=q_space_cube.shape[1])
    mask_empty = counts > 0
    q_space_cube[:, mask_empty] /= counts[mask_empty][None, :]

    return q_space_cube

### --- ###
def rescale_lambda_to_q(target_object, cube, lambdas, q_boundaries, 
                   m2lim=(0.017, 0.2), qlim=(0.05, 0.5), save_cols=True):

    # Precompute masses and q
    mass = target_object["mass"]
    mass_ratios = q_from_l_vectorized(lambdas, mass, target_object["parallax"])
    #companion_masses = mass * mass_ratios
    #print(mass, mass_ratios, companion_masses)

    # Mask of acceptable lambda points
    valid_mask = (
        #(companion_masses > m2lim[0]) &
        #(companion_masses < m2lim[1]) &
        #(mass_ratios > qlim[0]) &
        #(mass_ratios < qlim[1]) &
        (mass_ratios > 0)
    )

    # Get valid indices
    valid_idx = np.where(valid_mask)[0]

    # Bin companion masses
    col_bins = np.searchsorted(q_boundaries, mass_ratios[valid_idx], side="right")

    # Output array
    q_space_cube = np.zeros((cube.shape[0], len(q_boundaries) + 1), dtype=cube.dtype)

    # Add contributions column-wise
    np.add.at(q_space_cube, (slice(None), col_bins), cube[:, valid_idx])
    
    # we want the mean, so work out counts to divide by
    counts = np.bincount(col_bins, minlength=q_space_cube.shape[1])
    mask_empty = counts > 0
    q_space_cube[:, mask_empty] /= counts[mask_empty][None, :]
    
    # Fill any empty columns on the right by copying the last non-empty column
    if save_cols:
        last_valid = np.where(counts > 0)[0][-1]
        if sum(q_space_cube[:,-1]) == 0:  # last column is empty
            q_space_cube[:, last_valid+1:] = np.tile(q_space_cube[:, last_valid][:, np.newaxis], (1, q_space_cube.shape[1] - last_valid - 1)) # copy the nearest valid column

    return q_space_cube

### --- ###
def compute_grid(target_object, sc_cubes, period_boundaries, m_boundaries, q_space=True, mass_binned=False, use_mass_index=True, scale=5, plim=(1,8), save_cols=True):
    # the cube is stored with counts from the marginalisation
    # we need to divide out by this
    marg_counts = sc_cubes["meta"]["shape"][-1]
    
    plx_index = np.argmin(abs(sc_cubes["meta"]["parallaxes"] - target_object["parallax"]))
    reference_plx = sc_cubes["meta"]["parallaxes"][plx_index]
    
    # choose the right precomputed grid, with the right solution type
    if mass_binned:
        if use_mass_index == False:
            mass_index = np.argmin(abs(np.array(sc_cubes["meta"]["reference_masses"]) - target_object["mass"]))
        else:
            mass_index = target_object["mass_index"]
        reference_mass = sc_cubes["meta"]["reference_masses"][mass_index]
        working_cube = sc_cubes["data"][reference_mass][reference_plx][:,:,target_object["soltype_index"]]/marg_counts
        grid_lambdas = sc_cubes["meta"]["lambdas"][mass_index][plx_index]
    else:
        working_cube = sc_cubes["data"][reference_plx][:,:,target_object["soltype_index"]]/marg_counts
        grid_lambdas = sc_cubes["meta"]["lambdas"][plx_index]
    
    working_cube = scale_resolution(working_cube, scale=scale, axis=1)
    working_lambdas = np.linspace((grid_lambdas**(1/4))[0], (grid_lambdas**(1/4))[-1], scale*len(grid_lambdas))**4
    
    # scale it down to the right period binning
    period_scaled_cube = rescale_period(working_cube, np.log10(sc_cubes["meta"]["periods"]), period_boundaries, plim=plim)
    
    # and scale it to the working companion mass
    rescale_mass_coordinate = rescale_lambda
    if q_space:
        rescale_mass_coordinate = rescale_lambda_to_q
    fully_rescaled_cube = rescale_mass_coordinate(target_object, period_scaled_cube, working_lambdas, m_boundaries, save_cols=save_cols)
    
    return fully_rescaled_cube
        
### --- ###
def compute_grids(objects, sc_cubes, period_boundaries, m_boundaries, q_space=True, mass_binned=False, verbose=True, scale=5, plim=(1,8)):
    '''
        wrapper for compute_grid() (above)
    '''
    grids = []
    
    # for every object, first map to the scaled down period resolution
    # then, map it into m2-space, trimming the irrelevant m2s
    if verbose:
        pbar = tqdm(total=len(objects))
    for target_object in objects:
        fully_rescaled_cube = compute_grid(target_object, sc_cubes, period_boundaries, m_boundaries, 
                                           q_space=q_space, mass_binned=mass_binned, scale=scale, plim=plim)
        grids.append(fully_rescaled_cube.ravel())
        if verbose:
            pbar.update(1)
    
    return grids   

### --- ###
def gaussian(x, mu, sigma):
    '''
        this function evaluates a normalised gaussian at x, defined by two
        parameters: peak (mu) and width (sigma)
    '''
    return np.exp(-(mu - x)**2/(2*sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

### --- ###
def area_in_range(target_range, mu, sigma, resolution=100):
    xs = np.linspace(*target_range, resolution)
    ys = gaussian(xs, mu, sigma)
    return np.trapezoid(y=ys, x=xs)

### --- ###
def pexp(val, index, val_range=(0, 1), ignore_a=False):
    '''
        normalised power law probability
    '''
    a = 1
    if not ignore_a:
        a = (index + 1) / (val_range[1] ** (index + 1) - val_range[0] ** (index + 1))
    return a * (val ** index)

### -- ###
def area_in_range_powerlaw(target_range, index, resolution=100):
    xs = np.linspace(*target_range, resolution)
    ys = pexp(xs, index, ignore_a=True)
    return np.trapezoid(y=ys, x=xs)

### --- ###
def cutoff_to_fraction(p_model, pcut):
    p_mu, p_si = p_model
    total_area = area_in_range((1,pcut), p_mu, p_si, resolution=100)
    observable_area = area_in_range((2,3), p_mu, p_si, resolution=100)
    return observable_area / total_area

### --- ###
def fraction_to_cutoff(p_model, fraction):
    p_mu, p_si = p_model
    observable_area = area_in_range((2,3), p_mu, p_si, resolution=100)
    target_area = observable_area / fraction
    # search for cutoff
    pcut_vals = np.linspace(3,8,1000)
    for pcut in pcut_vals:
        total_area = area_in_range((1,pcut), p_mu, p_si, resolution=100)
        if total_area >= target_area:
            return pcut
    return 8.0

### --- ###
def create_model_cube(grid_shape, p_model=None, q_model=0, pcut=None,
                      p_range=(1,8), q_range=(0.05,0.5)):
    '''
        log-normal period distribution, power-law mass ratio distribution
        p_model: (T_mu, T_si)
        q_model: q^gamma power law index 
    '''
    
    # set up period distribution
    if p_model is None:
        p_dist = np.ones(grid_shape[0])/grid_shape[0]
    else:
        p_mu, p_si = p_model
        period_count = grid_shape[0]
        p_vals = np.linspace(*p_range, period_count+1)
        p_dist = np.zeros(period_count)
        total_area = area_in_range(p_range, p_mu, p_si, resolution=period_count*10)
        for i in range(period_count):
            p_dist[i] = area_in_range((p_vals[i],p_vals[i+1]), p_mu, p_si, resolution=period_count*10) / total_area
    
    # make an upper bound period cut
    if pcut is not None:
        p_dist[:np.argmin(abs(pcut[0]-np.linspace(*p_range, grid_shape[0])))] = 0 # lower cut
        p_dist[np.argmin(abs(pcut[1]-np.linspace(*p_range, grid_shape[0])))+1:] = 0 # upper cut
    
    # set up mass ratio distribution
    q_count = grid_shape[1]
    q_vals = np.linspace(*q_range,q_count+1)
    q_dist = np.zeros(q_count)
    total_area = area_in_range_powerlaw(q_range, q_model, resolution=q_count*10)
    for i in range(q_count):
        q_dist[i] = area_in_range_powerlaw((q_vals[i],q_vals[i+1]), q_model, resolution=q_count*10) / total_area
    
    # construct cube
    model_cube = np.outer(p_dist, q_dist)
    model_cube = model_cube / model_cube.sum() # should sum to 1 actually
    return model_cube

### ------------------- ###
### --- WD template --- ###

### --- ###
def f_q(q, ql, qh, b, m_rg, alpha):
    if (q < ql) | (q > qh):
        return 0
    return (q - b/m_rg)**(-1-alpha)

### --- ###
def calculate_ql(a, m_rg, b, d):
    return a*(m_rg+d)/m_rg + b/m_rg

### --- ###
def wd_create_model_cube(grid_shape, m_rg, a, b, d, alpha, 
                      p_model=None, pcut=None,
                      p_range=(1,8), q_range=(0.05,0.5)):
    # set up period distribution
    if p_model is None:
        p_dist = np.ones(grid_shape[0])/grid_shape[0]
    else:
        p_mu, p_si = p_model
        period_count = grid_shape[0]
        p_vals = np.linspace(*p_range, period_count+1)
        p_dist = np.zeros(period_count)
        total_area = area_in_range(p_range, p_mu, p_si, resolution=period_count*10)
        for i in range(period_count):
            p_dist[i] = area_in_range((p_vals[i],p_vals[i+1]), p_mu, p_si, resolution=period_count*10) / total_area
    
    # make bounded cuts on the period
    if pcut is not None:
        p_dist[:np.argmin(abs(pcut[0]-np.linspace(*p_range, grid_shape[0])))] = 0 # lower cut
        p_dist[np.argmin(abs(pcut[1]-np.linspace(*p_range, grid_shape[0])))+1:] = 0 # upper cut
    
    # set up mass ratio distribution
    q_count = grid_shape[1]
    q_vals = np.linspace(*q_range,q_count)

    ql = calculate_ql(a, m_rg, b, d)
    qh = 1.4/m_rg
    fqwds = np.array([f_q(q, ql, qh, b, m_rg, alpha) for q in q_vals])
    q_dist = fqwds / np.trapezoid(fqwds, q_vals)
    
    # construct cube
    model_cube = np.outer(p_dist, q_dist)
    model_cube = model_cube / model_cube.sum() # should sum to 1 actually
    return model_cube

### --- WD template --- ###
### ------------------- ###

def calculate_log_likelihood(fb, soltypes, grids, model_cube, cutoff=np.exp(-18)): 
    # compute individual solution chance
    dot_values = fb * np.dot(grids, model_cube.ravel())

    # For solution_type == 0, add single star component (1 - fb)
    dot_values[soltypes == 0] += (1 - fb)

    # Apply cutoff and sum log-likelihoods
    return np.sum(np.log(np.maximum(dot_values, cutoff)))

### --- ###
def within_prior(mcmc_params):
    fb, q_index = mcmc_params
    if (fb < 0) | (fb > 1):
        return False
    # if (pcut < 3) | (pcut > 8):
    #     return False
    if (q_index < -0.5) | (q_index > 3):
        return False
    return True

### --- ###
def wd_within_prior(mcmc_params):
    fb, fwd = mcmc_params
    if (fb < 0) | (fb > 1):
        return False
    if (fwd < 0) | (fwd > 1):
        return False
    return True

### --- ###
def likelihood_wrapper(mcmc_params, soltypes, grid_shape, p_model, pcut, cutoff):
    global _global_grids
    if not within_prior(mcmc_params):
        return -np.inf
    fb = mcmc_params[0]
    # pcut = 5 #mcmc_params[1]
    q_model = mcmc_params[1]

    model_cube = create_model_cube(grid_shape, p_model=p_model, q_model=q_model, pcut=pcut)
    return calculate_log_likelihood(fb, soltypes, _global_grids, model_cube, cutoff=cutoff)

### --- ###
def initialise_walkers(nwalkers):
    initial_params = np.zeros((nwalkers, 2))
    initial_params[:,0] = np.random.uniform(0.01,0.99, nwalkers) # fb
    #initial_params[:,1] = np.random.uniform(3,8, nwalkers)       # pcut
    initial_params[:,1] = np.random.uniform(0,3, nwalkers)       # q_index
    return initial_params

### --- ###
def wd_likelihood_wrapper(mcmc_params, wd_params, soltypes, grid_shape, p_model, q_model, pcut, cutoff):
    global _global_grids
    if not wd_within_prior(mcmc_params):
        return -np.inf
    fb, fwd = mcmc_params
    #fb = 0.6

    ms_model_cube = create_model_cube(grid_shape, p_model=p_model, q_model=q_model, pcut=pcut)
    wd_model_cube = wd_create_model_cube(grid_shape, *wd_params, p_model=p_model, pcut=pcut)
    model_cube = (1-fwd)*ms_model_cube + fwd*wd_model_cube
    return calculate_log_likelihood(fb, soltypes, _global_grids, model_cube, cutoff=cutoff)

### --- ###
def wd_initialise_walkers(nwalkers):
    initial_params = np.zeros((nwalkers, 2))
    initial_params[:,0] = np.random.uniform(0.01,0.99, nwalkers) # fb
    initial_params[:,1] = np.random.uniform(0.01,0.99, nwalkers) # fwd
    return initial_params

### --- ###
def fisher_uncertainty(likelihoods):
    fbs = np.linspace(0.02,0.98,len(likelihoods))
    dtheta = fbs[1] - fbs[0]
    ind_max = np.argmax(likelihoods)
    
    # compute second derivative from linear approximation
    second_derivative = np.array([likelihoods[ind_max + 1] - 2 * likelihoods[ind_max] + likelihoods[ind_max - 1]])/dtheta**2 
    # fisher information
    sigma = 1 / np.sqrt(-second_derivative)
    return sigma

### --- ###
class popsampler():
    def __init__(self, sc_cubes, catalogue, model_cube=None):
        self.sc_cubes = sc_cubes
        self.catalogue = catalogue
        
        # for imposed models
        self.model_cube = model_cube
        # for the results
        self.sampler = None
        self.fbs = None
        self.likelihood_set = None
        self.locals = None
        self.locals_ranges = None

    ### --- ###
    def constrain_parameters(self, p_model, pcut=(1,8), model_cube_shape=(35,25), step_count=1000, nwalkers=7,
                             p_range=(1,8), q_range=(0.05,0.5), cutoff=np.exp(-18), 
                            grids=None, catalogue=None, mass_binned=False, scale=5, verbose=True):
        if verbose:
            print("Reducing catalogue...")
        working_catalogue, soltypes = self.reduce_catalogue(catalogue=catalogue)
            
        # precompute the q-L mappings for all the objects
        if grids is None:
            if verbose:
                print("Computing grids...")
            grids = self.assign_grids(model_cube_shape, working_catalogue, p_range, q_range, mass_binned=mass_binned, scale=scale, verbose=verbose)
        
        # run mcmc
        print("Running markov chains...")
        args = (soltypes, grids, model_cube_shape, p_model, pcut, cutoff)
        self.sampler = None
        ndim = 2 # fb, q_index
        initial_params = initialise_walkers(nwalkers)
        pool = multiprocessing.Pool()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood_wrapper, 
                                    args=args, pool=pool)
        sampler.run_mcmc(initial_params, step_count, progress=True, skip_initial_state_check=True)
        print("Complete!")
        self.sampler = sampler
        self.locals = ["fb", "gamma"]
        self.locals_ranges = [(0,1), (-0.5,2)]
        
    ### --- ###
    def wd_constrain_parameters(self, wd_params, model_cube_shape=(35,25), p_model=(4, 1.3), q_model=0, pcut=(2,8), step_count=1000, nwalkers=5,
                             p_range=(1,8), q_range=(0.05,0.5), cutoff=np.exp(-18), 
                            grids=None, catalogue=None, mass_binned=False, scale=5, verbose=True):
        if verbose:
            print("Reducing catalogue...")
        working_catalogue, soltypes = self.reduce_catalogue(catalogue=catalogue)
            
        # precompute the q-L mappings for all the objects
        if grids is None:
            if verbose:
                print("Computing grids...")
            grids = self.assign_grids(model_cube_shape, working_catalogue, p_range, q_range, mass_binned=mass_binned, scale=scale, verbose=verbose)
        
        # run mcmc
        print("Running markov chains...")
        args = (wd_params, soltypes, model_cube_shape, p_model, q_model, pcut, cutoff)
        self.sampler = None
        ndim = 2 # fb, fwd
        initial_params = wd_initialise_walkers(nwalkers)
        pool = multiprocessing.Pool(initializer=set_global_grids, initargs=(grids,))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, wd_likelihood_wrapper, args=args, pool=pool)
        sampler.run_mcmc(initial_params, step_count, progress=True, skip_initial_state_check=True)
        print("Complete!")
        self.sampler = sampler
        self.locals = ["fb", "fwd"]
        self.locals_ranges = [(0,1), (0,1)]
    
    ### --- ###
    def q_along_grid(self, p_model, fb, pcut,
                             p_range=(1,8), q_range=(0.05,0.5), cutoff=np.exp(-18), 
                            grids=None, catalogue=None, model_cube=None, mass_binned=False, scale=5, verbose=True):
        temp_kwargs = dict()
        temp_kwargs["cutoff"] = cutoff
        temp_kwargs["pcut"] = pcut
        
        if verbose:
            print("Reducing catalogue...")
        working_catalogue, soltypes = self.reduce_catalogue(catalogue=catalogue)
        
        if model_cube is not None:
            self.model_cube = model_cube
            
        # precompute the q-L mappings for all the objects
        if grids is None:
            if verbose:
                print("Computing grids...")
            grids = self.assign_grids(working_catalogue, p_range, q_range, mass_binned=mass_binned, scale=scale, verbose=verbose)
        
        # run mcmc
        print("Calculating likelihoods...")
        args = (soltypes, grids, self.model_cube.shape, p_model)
        likelihoods = []
        gammas = np.linspace(-0.5,2,1000)
        for gamma in tqdm(gammas):
            likelihoods.append(likelihood_wrapper([fb,gamma], *args, **temp_kwargs))
            
        print("Complete!")
        return gammas, likelihoods
    
    def reduce_catalogue(self, catalogue=None, mass_binned=False):
        temp_catalogue = self.catalogue
        if catalogue is not None:
            temp_catalogue = catalogue
        working_catalogue = []
        soltypes = np.zeros(len(temp_catalogue), dtype=np.int8)
        for i, target_object in enumerate(temp_catalogue):
            # save just parallax and mass for grid conversion
            reduced_object = {
                "parallax": target_object["parallax"],
                "mass": target_object["mass"],
                "soltype_index": SOLUTION_TYPES.index(target_object["solution_type"])
            }
            if mass_binned:
                reduced_object["mass_index"] = target_object["mass_index"]
            working_catalogue.append(reduced_object)
            
            # save just solution type for use at inference
            soltypes[i] = target_object["solution_type"] 
        working_catalogue = np.array(working_catalogue)
        return working_catalogue, soltypes
    
    def assign_grids(self, target_shape, working_catalogue, p_range, q_range, mass_binned=False, scale=5, verbose=True):
        p_boundaries = np.linspace(*p_range, target_shape[0]+1)[1:-1]
        q_boundaries = np.linspace(*q_range, target_shape[1]+1)[1:-1]
        grids = np.array(compute_grids(working_catalogue, self.sc_cubes, p_boundaries, q_boundaries, 
                                       q_space=True, mass_binned=mass_binned, scale=scale, verbose=verbose))
        return grids
    
    def binarity(self, resolution=250, p_range=(1,8), q_range=(0.05,0.5), cutoff=np.exp(-18), 
                 grids=None, catalogue=None, model_cube=None, mass_binned=False, scale=5, verbose=True):
        '''
            binarity likelihood across fb
        '''        
        if verbose:
            print("Reducing catalogue...")
        working_catalogue, soltypes = self.reduce_catalogue(catalogue=catalogue, mass_binned=mass_binned)
        
        working_model_cube = self.model_cube
        if model_cube is not None:
            working_model_cube = model_cube
        # precompute the q-L mappings for all the objects
        if grids is None:
            if verbose:
                print("Computing grids...")
            grids = self.assign_grids(working_model_cube.shape, working_catalogue, p_range, q_range, mass_binned=mass_binned, scale=scale, verbose=verbose)
        
        if verbose:
            print("Computing likelihoods...")
        fbs = np.linspace(0.02,0.98,resolution)
        likelihoods = np.zeros(resolution)
        if verbose:
            pbar = tqdm(total=resolution)
        for i in range(resolution):
            likelihoods[i] = calculate_log_likelihood(fbs[i], soltypes, grids, working_model_cube, cutoff=cutoff)
            if verbose:
                pbar.update(1)

        self.fbs = fbs
        self.likelihood_set = likelihoods
        return fbs, likelihoods
    
    def binarity_precomputations(self, p_range=(1,8), q_range=(0.05,0.5)):
        working_catalogue, _ = self.reduce_catalogue()
        grids = self.assign_grids(working_catalogue, p_range, q_range)
        return working_catalogue, grids
        
    def binarity_binned_mass(self, model_cube, working_catalogue, grids, mass_lims, p_range=(1,8), q_range=(0.05,0.5), **kwargs):        
        constraining_results = np.zeros((len(mass_lims), 3))
        for i in tqdm(range(len(mass_lims))):
            temp_grids = []
            temp_catalogue = []
            for j, obj in enumerate(working_catalogue):
                obj["solution_type"] = [0,5,7,9,12][obj["soltype_index"]]
                if i == 0:
                    if obj["mass"] < mass_lims[0]:
                        temp_catalogue.append(obj)
                        temp_grids.append(grids[j])
                else:
                    if (mass_lims[i-1] < obj["mass"]) & (obj["mass"] < mass_lims[i]):
                        temp_catalogue.append(obj)
                        temp_grids.append(grids[j])
            self.binarity(p_range=p_range, q_range=q_range, grids=temp_grids, catalogue=temp_catalogue, model_cube=model_cube, verbose=False, **kwargs)
            constraining_results[i] = self.fb_estimator()
        return constraining_results
    
    def fb_estimator(self, cut=2, results=None):
        '''
            return MLE with pm
            cut: number of sigmas that the pm should represent, defaults to 2
            results: (fbs, likelihoods) tuple of lists of equal size corresponding
                to the sampled binary fractions and their corresponding likelihoods
        '''
        working_ls, working_fbs = self.likelihood_set, self.fbs
        if results is not None:
            working_ls, working_fbs = results
        working_ls, working_fbs = np.array(working_ls), np.array(working_fbs)
        working_ls -= np.max(working_ls)
        
        peakdx = np.argmax(working_ls)
        maximum = working_fbs[peakdx]    
        minus_loc = working_fbs[:peakdx][np.argmin(abs(working_ls[:peakdx]+cut))]   
        plus_loc = working_fbs[peakdx:][np.argmin(abs(working_ls[peakdx:]+cut))]  
        return maximum, plus_loc-maximum, maximum-minus_loc
    
    def fb_likelihood(self, fb=None, name=None, **kwargs):
        plt.clf();
        mle, p, m  = self.fb_estimator()
        plt.plot(self.fbs, self.likelihoods, c="black", linewidth=3, **kwargs);
        plt.axvline(x=mle, c="red", linestyle="--", label=r'FIT: ${:.3f}^{{+{:.3f}}}_{{-{:.3f}}}$'.format(mle,p,m));
        if fb is not None:
            plt.axvline(x=fb, c="green", linestyle="--", label=f"TRUTH: {fb}")
        plt.xlabel("binary fraction");
        plt.ylabel("log-likelihood");
        plt.legend();
        if name is not None:
            plt.title(name)
        plt.show();
        
    ### --- ###
    def chain(self, discard=25):
        return self.sampler.get_chain(discard=discard, flat=True)
    
    ### --- ###
    def likelihoods(self, discard=25):
        return self.sampler.get_log_prob(discard=discard, flat=True)       
    
    ### --- ###
    def apply_condition(self, condition, chain, likelihoods):
        set_locals = {self.locals[i]:chain[:,i] for i in range(len(self.locals))}
        set_locals["likelihood"] = likelihoods
        ran_condition = eval(condition, set_locals)
        
        chain = chain[ran_condition]
        total = len(likelihoods)
        likelihoods = likelihoods[ran_condition]
        if self.verbose:
            print("{}/{} ({:.1f}%) of sampled points remain".format(len(likelihoods), total, len(likelihoods)/total*100))
        
        return chain, likelihoods
    
    ### --- ###
    def plot_corner(self, condition=None, discard=25, full_prior=False, **kwargs):
        chain, likelihoods = self.chain(discard=discard), self.likelihoods(discard=discard)
        
        if condition is not None:
            chain, likelihoods = self.apply_condition(condition, chain, likelihoods)

        ranges = [(param.min(), param.max()) if np.ptp(param) > 0 else (param[0]-1e-3, param[0]+1e-3) for param in chain.T]
        
        if full_prior:
            ranges = self.locals_ranges

        return corner.corner(chain, ranges=ranges, labels=self.locals, **kwargs);
    
    ### --- ###
    def plot_2d(self, parameters, condition=None, truths=None, savedir=None, discard=25, full_prior=False, **kwargs):
        chain, likelihoods = self.chain(discard=discard), self.likelihoods(discard=discard)
        
        if condition is not None:
            chain, likelihoods = self.apply_condition(condition, chain, likelihoods)

        check_indices = [self.locals.index(param) for param in parameters]

        fig, ax = plt.subplots(1,1)
        
        cb = ax.scatter(chain[:,check_indices[0]], chain[:,check_indices[1]], c=likelihoods, cmap='viridis', norm=colors.Normalize(), **kwargs)
        plt.colorbar(cb, label="log likelihood")
        ax.set_xlabel(self.locals[check_indices[0]]);
        ax.set_ylabel(self.locals[check_indices[1]]);
        max_x, max_y = chain[:,check_indices[0]][np.argmax(likelihoods)], chain[:,check_indices[1]][np.argmax(likelihoods)]
        ax.axvline(max_x, c="k", linestyle="--", label="highest likelihood");
        ax.axhline(max_y, c="k", linestyle="--");
        
        if truths is not None:
            ax.axvline(truths[0], c="r", linestyle="--", label="truth");
            ax.axhline(truths[1], c="r", linestyle="--");

        ax.legend()
        
        if full_prior:
            ax.set_xlim(self.locals_ranges[self.locals[check_indices[0]]])
            ax.set_ylim(self.locals_ranges[self.locals[check_indices[1]]])
        
        if savedir is not None:
            plt.savefig(savedir)
            
        return fig
    
    ### --- ###
    def plot_parameter(self, parameter, truth=None, condition=None, discard=25):
        chain, likelihoods = self.chain(discard=discard), self.likelihoods(discard=discard)
        
        if condition is not None:
            chain, likelihoods = self.apply_condition(condition, chain, likelihoods)
        
        check_index = list(self.locals).index(parameter)
        
        plt.plot(chain[:,check_index], color="maroon")
        if truth is not None:
            plt.axhline(y=truth, c="k", linestyle="--")
        plt.title(parameter)
        
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