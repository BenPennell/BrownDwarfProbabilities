import numpy as np
import emcee
import multiprocessing
import datetime
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import corner
from OldSampler import q_from_l_vectorized
from utils.utils import *

try:
    # for Jupyter
    from tqdm.notebook import tqdm
except ImportError:
    # for terminal
    from tqdm import tqdm

# to save on memory, make a global variable for the grids
_global_grids = None
_global_model_cubes = None

def set_global_grids(grids):
    global _global_grids
    _global_grids = grids

def set_global_model_cubes(model_cubes):
    global _global_model_cubes
    _global_model_cubes = model_cubes


SOLUTION_TYPES = [0,5,7,9,12]

def scale_resolution(arr, scale=2, axis=0, even=False):
    """
    Upscale a numpy array along a given axis by repeating values.
    If even=True, divide repeated values by scale to preserve total sum.
    """
    arr = np.asarray(arr)
    expanded = np.expand_dims(arr, axis + 1)
    repeated = np.repeat(expanded, scale, axis=axis + 1)
    if even:
        repeated = repeated / scale
    new_shape = list(arr.shape)
    new_shape[axis] *= scale
    return repeated.reshape(new_shape)

### --- ###
def rescale_period(cube, periods, period_boundaries):
    """
    Rescale a 2D array along the period axis using binning.
    
    Parameters
    ----------
    cube : np.ndarray
        Shape (n_periods, n_lambdas)
    periods : np.ndarray
        Centers of the original period bins
    period_boundaries : array_like
        Edges of the desired period bins (length N+1)
        
    Returns
    -------
    reshaped_cube : np.ndarray
        Shape (len(period_boundaries)-1, n_lambdas)
    """
    cube = np.asarray(cube)
    
    # Bin periods
    bin_indices = np.searchsorted(period_boundaries, periods, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(period_boundaries)-2)  # ensure within 0..N-1
    
    # Sum values into new bins
    reshaped_cube = np.zeros((len(period_boundaries)-1, cube.shape[1]), dtype=cube.dtype)
    np.add.at(reshaped_cube, (bin_indices, slice(None)), cube)
    
    # Compute counts per bin and normalize
    counts = np.bincount(bin_indices, minlength=reshaped_cube.shape[0])
    mask = counts > 0
    reshaped_cube[mask] /= counts[mask][:, None]
    
    return reshaped_cube

### --- ###
def rescale_lambda_to_q(target_object, cube, grid_lambdas, q_boundaries, column_areas=None, qlim=(0.05, 0.5), save_cols=True):
    """
        Rescale a lambda-spaced cube into q-space
        
        grid_lambdas: the centres of each lambda bin (IN LAMBDA SPACE, NOT LAMBDA**(1/4))
        
        The way we do this is by creating a a grid with the q-spacing that we like, and then summing together all the columns
        from the original cube that would correspond to the same column in the output cube.
    
        period_boundaries, m_boundaries each correspond to (N+1,M+1) if the desired grid size is (N,M) since they represent
        the edges of the bins (so like how you'd specify it for a histogram)
    """
    # construct the output array to be the desired size
    q_space_cube = np.zeros((cube.shape[0], len(q_boundaries)-1), dtype=cube.dtype)
    
    # first, use the object's parameters to determine what the lambda spacing corresponds to in q-space
    mass, plx = target_object["mass"], target_object["parallax"]
    mass_ratios = q_from_l_vectorized(grid_lambdas, mass, plx)
    
    # restrict the lambdas to the mass ratios in the desired regime
    valid_mask = (mass_ratios >= qlim[0]) & (mass_ratios <= qlim[1])
    valid_idx = np.where(valid_mask)[0]
    
    # check which column in the output cube each q-space grid point corresponds to
    col_bins = np.searchsorted(q_boundaries, mass_ratios[valid_idx], side="right") - 1
    col_bins = np.clip(col_bins, 0, len(q_boundaries)-2)  # shouldn't need this, but it makes sure the index is valid.
    
    # Sum up each column of the original cube that corresponds to the same column in the output cube
    # weight it by the area of the original column
    if column_areas is None:
        column_areas = np.ones_like(grid_lambdas[:-1]) # if no column areas are provided, just treat them as 1 (i.e. don't preserve probability mass when splitting up lambda bins)
    np.add.at(q_space_cube, (slice(None), col_bins), cube[:, valid_idx] * column_areas[valid_idx])
    
    # get the total area that went in to each bin, to be divided out at the end
    weight_sums = np.bincount(col_bins, weights=column_areas[valid_idx], minlength=q_space_cube.shape[1])
    
    # Since this is not a probability density, we don't want the "total probability" of being in a particular grid point
    # instead, this cube is asking what is the probability that a particular grid point corresponds to a particular solution type
    # for that reason, we average over the columns
    # this is to account for the chance taht two columns in l-space which have different probabilities are mapped to the same column in q-space
    # so, the new probabilities in that q-space column should be the average of the two
    # and, it should be the weighted average of the columns. We already added them up scaled by the area, and then we should divide by that total area
    # this achieves the desired weighted-average effect
    mask_nonempty = weight_sums > 0
    q_space_cube[:, mask_nonempty] /= weight_sums[mask_nonempty][None, :] # divide out by the cound of lambda-columns
    
    # Finally: if there is no lambda columns corresponding to the highest q-space columns,
    # this was just a problem in the precomputing, we didn't make the grid wide enough.
    # In this case, if the highest q-columns are zero (which they should never be), we can
    # fill from the last valid column, since the probabilities are monotonic in q and are 
    # quite flat at the highest mass ratios. This is an okay approximation, especially
    # since this should only come up for a few objects which conspire to be super close+massive
    if save_cols:
        last_valid = None
        for i in range(q_space_cube.shape[1]):
            if weight_sums[i] > 0:
                last_valid = i
            elif last_valid is not None:
                q_space_cube[:, i] = q_space_cube[:, last_valid]
    
    return q_space_cube

### --- ###
def compute_grid(target_object, sc_cubes, period_boundaries, m_boundaries, scale=5, save_cols=True):
    '''
        The precomputed grid needs to be mapped from lambda-space into q-space using an object's parallax and primary mass.
        Plus, it would be nice to rescale the grid to a target resolution. Both are done in this function
        
        Crucially: the lambas come evenly spaced in lambda**(1/4), so we need to be careful of the jacobian
        
        period_boundaries, m_boundaries each correspond to (N+1,M+1) if the desired grid size is (N,M) since they represent
        the edges of the bins (so like how you'd specify it for a histogram)
    '''
    
    # the cube is stored with counts from the marginalisation
    # we need to divide out by this
    marg_counts = sc_cubes["meta"]["shape"][-1]
    
    # choose the right precomputed grid, with the right solution type
    plx_index = np.argmin(abs(sc_cubes["meta"]["plx_grid"] - target_object["parallax"]))
    #mass_index = np.argmin(abs(np.array(sc_cubes["meta"]["mass_grid"]) - target_object["mass"])) 
    mass_index = target_object["mass_index"]
    working_cube = sc_cubes["hist"][mass_index][plx_index][:,:,target_object["soltype_index"]]/marg_counts # pull out the right grid
    grid_lambdas = sc_cubes["meta"]["lambdas"][mass_index][plx_index] # and extract the correspondings lambdas for this grid
    
    # we need to convert into q space, using the object's parallax and mass
    # we will try to avoid problems that arise from large discrete lambda bins (such as empty bins in q space) by
    # first scaling up and splitting the lambda bins to simulate a finer resolution 
    working_cube = scale_resolution(working_cube, scale=scale, axis=1) # puff up the lambda^(1/4) axis by scale=5 times
    working_lambda_spacings = np.linspace((grid_lambdas**(1/4))[0], (grid_lambdas**(1/4))[-1], scale*(len(grid_lambdas)-1)+1) # in lambda^(1/4) space
    working_lambdas = working_lambda_spacings**4 # in lambda space
    lambda_centres = working_lambdas[:-1] + (working_lambdas[1:] - working_lambdas[:-1])/2 # the centers of the lambda bins, to be used for the q-lambda conversion
    column_areas = working_lambdas[1:] - working_lambdas[:-1] # the widths of the lambda bins, to preserve probability mass when we split them up
    
    # The user supplied a particular desired period resolution, remap the grid to this resolution
    working_periods = np.log10(sc_cubes["meta"]["periods"])
    period_centers = working_periods[:-1] + (working_periods[1:]-working_periods[:-1])/2
    period_scaled_cube = rescale_period(working_cube, period_centers, period_boundaries)
    
    # Finally, convert to mass ratio space and restrict to the supplied mass ratio resolution
    fully_rescaled_cube = rescale_lambda_to_q(target_object, period_scaled_cube, lambda_centres, m_boundaries, column_areas=column_areas, save_cols=save_cols)
    
    return fully_rescaled_cube

# ----------------------
def compute_grids(objects, sc_cubes, period_boundaries, q_boundaries, scale=5, verbose=True):
    grids = []
    if verbose:
        pbar = tqdm(total=len(objects))
    for target in objects:
        grid = compute_grid(target, sc_cubes, period_boundaries, q_boundaries, scale=scale)
        grids.append(grid.ravel())
        if verbose:
            pbar.update(1)
    return np.array(grids)

# ----------------------
def compute_grids_merged(objects, sc_cubes_small, sc_cubes_big, period_boundaries, q_boundaries, scale=5, verbose=True):
    grids = []
    if verbose:
        pbar = tqdm(total=len(objects))
    for target in objects:
        if target["mass"] < 0.3:
            sc_cubes = sc_cubes_small
        else:
            sc_cubes = sc_cubes_big
        grid = compute_grid(target, sc_cubes, period_boundaries, q_boundaries, scale=scale)
        grids.append(grid.ravel())
        if verbose:
            pbar.update(1)
    return np.array(grids)
    
### --- ###
def full_rescale_lambda_to_q(target_object, cube, lambdas, q_boundaries,
                       m2lim=(0.017, 0.2), qlim=(0.05, 0.5), save_cols=True):

    # cube shape: (P, L, S)
    P, L, S = cube.shape

    mass = target_object["mass"]
    mass_ratios = q_from_l_vectorized(lambdas, mass, target_object["parallax"])  # (L,)

    valid_mask = (
    #(companion_masses > m2lim[0]) &
    #(companion_masses < m2lim[1]) &
    #(mass_ratios > qlim[0]) &
    (mass_ratios < qlim[1]) &
    (mass_ratios > 0)
    )
    valid_idx = np.where(valid_mask)[0]

    # bin indices for valid lambda points
    col_bins = np.searchsorted(q_boundaries, mass_ratios[valid_idx], side="left")  # (Nv,)

    # output: (P, Q, S)
    Q = len(q_boundaries) + 1
    q_space_cube = np.zeros((P, Q, S), dtype=cube.dtype)

    # accumulate
    # cube[:, valid_idx, :] → (P, Nv, S)
    np.add.at(q_space_cube, (slice(None), col_bins, slice(None)), cube[:, valid_idx, :])

    # counts per bin
    counts = np.bincount(col_bins, minlength=Q)
    mask = counts > 0

    # divide (broadcast over P and S)
    q_space_cube[:, mask, :] /= counts[mask][None, :, None]

    if save_cols:
        valid_cols = np.where(counts > 0)[0]
        if len(valid_cols) > 0:
            last_valid = valid_cols[-1]

            # check if last column is empty
            if np.all(q_space_cube[:, -1, :] == 0):
                q_space_cube[:, last_valid+1:, :] = q_space_cube[:, last_valid:last_valid+1, :]

    return q_space_cube

### --- ###
def full_rescale_period(cube, periods, period_boundaries, plim=(1, 8)):
    # cube: (P, L, S)
    P, L, S = cube.shape

    bin_indices = np.searchsorted(period_boundaries, periods, side="right")  # (P,)

    # output: (P_new, L, S)
    P_new = len(period_boundaries) + 1
    reshaped_cube = np.zeros((P_new, L, S), dtype=cube.dtype)

    # accumulate across period axis
    np.add.at(reshaped_cube, (bin_indices, slice(None), slice(None)), cube)

    # counts per bin
    counts = np.bincount(bin_indices, minlength=P_new)
    mask = counts > 0

    reshaped_cube[mask, :, :] /= counts[mask, None, None]

    return reshaped_cube

### --- ###
def compute_grid_all_soltypes(target_object, sc_cubes, period_boundaries, m_boundaries,
                              q_space=True, scale=5, plim=(1, 8), save_cols=True):
    # the cube is stored with counts from the marginalisation
    marg_counts = sc_cubes["meta"]["shape"][-1]

    # choose the right precomputed grid
    plx_index = np.argmin(abs(sc_cubes["meta"]["plx_grid"] - target_object["parallax"]))
    mass_index = target_object["mass_index"]

    # keep ALL soltypes (no indexing on last axis)
    working_cube = sc_cubes["hist"][mass_index][plx_index] / marg_counts
    grid_lambdas = sc_cubes["meta"]["lambdas"][mass_index][plx_index]

    # working_cube shape is now: (period, lambda, soltype)

    # scale lambda resolution (axis=1 still correct)
    working_cube = scale_resolution(working_cube, scale=scale, axis=1)

    working_lambdas = np.linspace(
        (grid_lambdas**(1/4))[0],
        (grid_lambdas**(1/4))[-1],
        scale * (len(grid_lambdas) - 1) + 1
    )**4

    lambda_centers = working_lambdas[:-1] + (working_lambdas[1:] - working_lambdas[:-1]) / 2

    # period scaling
    working_periods = np.log10(sc_cubes["meta"]["periods"])
    period_centers = working_periods[:-1] + (working_periods[1:] - working_periods[:-1]) / 2

    # IMPORTANT: rescale_period must support extra trailing dimension
    period_scaled_cube = full_rescale_period(
        working_cube, period_centers, period_boundaries, plim=plim
    )

    # choose mass coordinate transform
    rescale_mass_coordinate = full_rescale_lambda_to_q if q_space else rescale_lambda

    # IMPORTANT: this function must also support broadcasting over soltype axis
    fully_rescaled_cube = rescale_mass_coordinate(
        target_object,
        period_scaled_cube,
        lambda_centers,
        m_boundaries,
        save_cols=save_cols
    )

    return fully_rescaled_cube 

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

def mask_and_renormalize_model_cube(model_cube, objects, q_boundaries):
    """
    Mask and renormalize a model cube per object using fractional bin overlap in q.

    Parameters
    ----------
    model_cube : np.ndarray
        Shape (n_period_bins, n_q_bins)
    objects : list of dict
        Each must have 'q_max'
    q_boundaries : array_like
        Bin edges (length n_q_bins + 1)

    Returns
    -------
    model_cubes_flat : np.ndarray
        Shape (n_objects, n_period_bins * n_q_bins)
    """
    n_period, n_q = model_cube.shape
    n_objects = len(objects)

    # Precompute bin widths
    q_low = q_boundaries[:-1]
    q_high = q_boundaries[1:]
    bin_widths = q_high - q_low

    base_cube = model_cube.astype(float)
    model_cubes_flat = np.zeros((n_objects, n_period * n_q))

    for i, obj in enumerate(objects):
        qmax = obj["q_max"]

        # Compute fractional overlap for each q bin
        overlap = np.clip(np.minimum(q_high, qmax) - q_low, 0, bin_widths)
        fractions = overlap / bin_widths  # between 0 and 1

        # Apply fractions to cube
        scaled_cube = base_cube * fractions[None, :]

        # Renormalize
        total = scaled_cube.sum()
        if total > 0:
            scaled_cube /= total

        model_cubes_flat[i] = scaled_cube.ravel()

    return model_cubes_flat

### --- ###
def calculate_log_likelihood(fb, soltypes, cutoff=np.exp(-30)): 
    global _global_grids, _global_model_cubes
    # compute individual solution chance
    dot_values = fb * np.sum(_global_grids * _global_model_cubes, axis=1)

    # For solution_type == 0, add single star component (1 - fb)
    dot_values[soltypes == 0] += (1 - fb)
    
    # Apply cutoff and sum log-likelihoods
    return np.sum(np.log(np.maximum(dot_values, cutoff)))

### --- ###
def expected_counts(fb, grids, model_cube): 
    ravelled_model_cube = model_cube.ravel()
    
    grids = grids.reshape((len(grids), len(ravelled_model_cube), 5)) # (#objects, grid_size, 5)
    grids = np.swapaxes(grids, 1, 2)
    
    dot_values = fb * np.dot(grids, ravelled_model_cube) # (#objects, 5)

    # For solution_type == 0, add single star component (1 - fb)
    dot_values[:,0] += (1 - fb)
    
    # Apply cutoff and sum log-likelihoods
    return dot_values #np.sum(dot_values, axis=0)

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
    def __init__(self, sc_cubes, catalogue, backup_sc=None, model_cube=None):
        self.sc_cubes = sc_cubes
        self.backup_sc = backup_sc
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
                             p_range=(1,8), q_range=(0.05,0.5), cutoff=np.exp(-30), 
                            grids=None, catalogue=None, scale=5, verbose=True):
        if verbose:
            print("Reducing catalogue...")
        working_catalogue, soltypes = self.reduce_catalogue(catalogue=catalogue)
            
        # precompute the q-L mappings for all the objects
        if grids is None:
            if verbose:
                print("Computing grids...")
            grids = self.assign_grids(model_cube_shape, working_catalogue, p_range, q_range, scale=scale, verbose=verbose)
        
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
    def q_along_grid(self, p_model, fb, pcut,
                             p_range=(1,8), q_range=(0.05,0.5), cutoff=np.exp(-30), 
                            grids=None, catalogue=None, model_cube=None, scale=5, verbose=True):
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
            grids = self.assign_grids(working_catalogue, p_range, q_range, scale=scale, verbose=verbose)
        
        # run mcmc
        print("Calculating likelihoods...")
        args = (soltypes, grids, self.model_cube.shape, p_model)
        likelihoods = []
        gammas = np.linspace(-0.5,2,1000)
        for gamma in tqdm(gammas):
            likelihoods.append(likelihood_wrapper([fb,gamma], *args, **temp_kwargs))
            
        print("Complete!")
        return gammas, likelihoods
    
    def reduce_catalogue(self, catalogue=None):
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
                "phot_g_mean_mag": target_object["phot_g_mean_mag"],
                "q_max": target_object["q_max"],
                "mass_index": target_object["mass_index"],
                "soltype_index": SOLUTION_TYPES.index(target_object["solution_type"])
            }
            working_catalogue.append(reduced_object)
            
            # save just solution type for use at inference
            soltypes[i] = target_object["solution_type"] 
        working_catalogue = np.array(working_catalogue)
        return working_catalogue, soltypes
    
    def assign_grids(self, target_shape, working_catalogue, p_range, q_range, scale=5, verbose=True):
        '''
            DO YOU ACTUALLY ENED THE BACKUP SC AND THE MERGED COMPUTE GRIDS?? CHECK
        '''
        p_boundaries = np.linspace(*p_range, target_shape[0]+1)
        q_boundaries = np.linspace(*q_range, target_shape[1]+1)
        grids = np.array(compute_grids(working_catalogue, self.sc_cubes, p_boundaries, q_boundaries, 
                                       scale=scale, verbose=verbose))
        # grids = np.array(compute_grids_merged(working_catalogue, self.sc_cubes, self.backup_sc, p_boundaries, q_boundaries, 
        #                                scale=scale, verbose=verbose))
        return grids
    
    def binarity(self, resolution=250, p_range=(1,8), q_range=(0.05,0.5), cutoff=np.exp(-30), 
                 grids=None, catalogue=None, model_cube=None, scale=5, verbose=True):
        '''
            binarity likelihood across fb
        '''        
        if verbose:
            print("Reducing catalogue...")
        working_catalogue, soltypes = self.reduce_catalogue(catalogue=catalogue)
        
        #effective_volumes = np.array([relative_volume(working_catalogue[i]["phot_g_mean_mag"], working_catalogue[i]["parallax"]) for i in range(len(working_catalogue))])
        #effective_volumes = generate_rolling_average(working_catalogue)
        
        working_model_cube = self.model_cube
        if model_cube is not None:
            working_model_cube = model_cube
            
        # precompute the q-L mappings for all the objects
        if grids is None:
            if verbose:
                print("Computing grids...")
            grids = self.assign_grids(working_model_cube.shape, working_catalogue, p_range, q_range, scale=scale, verbose=verbose)
        
        if verbose:
            print("Computing likelihoods...")
            pbar = tqdm(total=resolution)
        
        model_cubes = mask_and_renormalize_model_cube(working_model_cube, working_catalogue, np.linspace(*q_range, working_model_cube.shape[1]+1))
        
        set_global_grids(grids)
        set_global_model_cubes(model_cubes)
        
        fbs = np.linspace(0.02,0.98,resolution)
        likelihoods = np.zeros(resolution)
        for i in range(resolution):
            likelihoods[i] = calculate_log_likelihood(fbs[i], soltypes, cutoff=cutoff)
            if verbose:
                pbar.update(1)

        self.fbs = fbs
        self.likelihood_set = likelihoods
        return fbs, likelihoods
        
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
            cut: number of sigmas*2 that the pm should represent, defaults to 2 (1 sigma)
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