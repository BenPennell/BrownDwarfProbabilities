from utils.dependencies import *;
import emcee
import multiprocessing
import corner

SOLUTION_TYPES = [0,5,7,9,12]

def scale_resolution(arr, scale=2, axis=0, even=False):
    """
    Upscale a numpy array along a given axis by repeating values.
    If even=True, evenly divide repeated values by the scale to preserve total sum.
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
    n_bins, n_lambda = len(period_boundaries) - 1, cube.shape[1]

    reshaped_cube = np.zeros((n_bins, n_lambda), dtype=cube.dtype)

    # vectorized scatter-add 
    row_idx = np.repeat(bin_indices, n_lambda)
    col_idx = np.tile(np.arange(n_lambda), len(bin_indices))

    np.add.at(reshaped_cube, (row_idx, col_idx), cube.ravel())

    # normalize
    counts = np.bincount(bin_indices, minlength=n_bins)
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
def compute_grid(target_object, sc_cube, period_boundaries, m_boundaries, scale=5, save_cols=True, combine_higher_order=False):
    '''
        The precomputed grid needs to be mapped from lambda-space into q-space using an object's parallax and primary mass.
        Plus, it would be nice to rescale the grid to a target resolution. Both are done in this function
        
        Crucially: the lambas come evenly spaced in lambda**(1/4), so we need to be careful of the jacobian
        
        period_boundaries, m_boundaries each correspond to (N+1,M+1) if the desired grid size is (N,M) since they represent
        the edges of the bins (so like how you'd specify it for a histogram)
    '''
    
    # the cube is stored with counts from the marginalisation
    # we need to divide out by this
    soltype_index = SOLUTION_TYPES.index(target_object["solution_type"])
    marg_counts = sc_cube["meta"]["shape"][-1]
    # pull out the right grid for the solution type
    if combine_higher_order and soltype_index > 0:
        working_cube = sc_cube["cube"][:,:,1:].sum(axis=2)/marg_counts # pull out the right grids for the solution types, and sum them together
    else:
        working_cube = sc_cube["cube"][:,:,soltype_index]/marg_counts 
    grid_lambdas = sc_cube["meta"]["grid"][1] # and extract the correspondings lambdas for this grid, this is stored as LAMBDA EDGES (m+1)
    
    # we need to convert into q space, using the object's parallax, mass, and apparent magnitude
    # we will try to avoid problems that arise from large discrete lambda bins (such as empty bins in q space) by
    # first scaling up and splitting the lambda bins to simulate a finer resolution 
    working_cube = scale_resolution(working_cube, scale=scale, axis=1) # puff up the lambda^(1/4) axis by scale=5 times
    working_lambda_spacings = np.linspace((grid_lambdas**(1/4))[0], (grid_lambdas**(1/4))[-1], scale*(len(grid_lambdas)-1)-scale) # in lambda^(1/4) space
    working_lambdas = working_lambda_spacings**4 # in lambda space
    # we need the measurement uncertainty ratio to convert the lambdas
    uncertainty_ratio = al_uncertainty_per_ccd_interp(sc_cube["meta"]["reference_magnitude"]) / al_uncertainty_per_ccd_interp(target_object["phot_g_mean_mag"])
    # the new lambdas are "effectively smaller" when the uncertainty is higher than the reference uncertainty
    working_lambdas = working_lambdas / uncertainty_ratio
    
    lambda_centres = working_lambdas[:-1] + (working_lambdas[1:] - working_lambdas[:-1])/2 # the centers of the lambda bins, to be used for the q-lambda conversion
    column_areas = working_lambdas[1:] - working_lambdas[:-1] # the widths of the lambda bins, to preserve probability mass when we split them up
    
    # The user supplied a particular desired period resolution, remap the grid to this resolution
    period_centers = sc_cube["meta"]["period_grid"] # this is the CENTRES, (n)
    period_scaled_cube = rescale_period(working_cube, period_centers, period_boundaries)
    
    # Finally, convert to mass ratio space and restrict to the supplied mass ratio resolution
    fully_rescaled_cube = rescale_lambda_to_q(target_object, period_scaled_cube, lambda_centres, m_boundaries, column_areas=column_areas, save_cols=save_cols)
    
    return fully_rescaled_cube

# ----------------------
def compute_grids(objects, sc_cubes, period_boundaries, q_boundaries, scale=5, verbose=True, combine_higher_order=False):
    grids = []
    iterator = tqdm(objects, desc="Computing grids") if verbose else objects
    for target in iterator:
        grid = compute_grid(target, sc_cubes, period_boundaries, q_boundaries, scale=scale, combine_higher_order=combine_higher_order)
        grids.append(grid.ravel())
    return np.array(grids)

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
    
    # apply cuts in period if desired
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

    # Precompute bin widths
    q_low = q_boundaries[:-1]
    q_high = q_boundaries[1:]
    bin_widths = q_high - q_low

    base_cube = np.array(model_cube, copy=True)
    model_cubes_flat = np.zeros((len(objects), n_period * n_q))

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
def calculate_log_likelihood(grids, model_cubes, solution_types_arr, fb, cutoff=np.exp(-30)):
    # compute individual solution chance
    dot_values = fb * np.sum(grids * model_cubes, axis=1)

    # For solution_type == 0, add single star component (1 - fb)
    dot_values[solution_types_arr == 0] += (1 - fb)
    
    # Apply cutoff and sum log-likelihoods
    return np.sum(np.log(np.clip(dot_values, cutoff, None)))

### --- ###
def fisher_uncertainty(likelihoods):
    fb_grid = np.linspace(0.02,0.98,len(likelihoods))
    dtheta = fb_grid[1] - fb_grid[0]
    ind_max = np.argmax(likelihoods)
    
    # compute second derivative from linear approximation
    second_derivative = np.array([likelihoods[ind_max + 1] - 2 * likelihoods[ind_max] + likelihoods[ind_max - 1]])/dtheta**2 
    # fisher information
    sigma = 1 / np.sqrt(-second_derivative)
    return sigma

def multiplicity(catalogue, model_cube, sc_cube, resolution=1000, p_range=(1,8), q_range=(0.05,0.5), 
                 cutoff=np.exp(-30), grids=None, scale=5, verbose=True, combine_higher_order=False):
    '''
        Likelihood for all binary fractions for a given choice of model on a given set of objects
        
        catalogue: should be an astropy table with the columns "parallax", "mass", "phot_g_mean_mag", "q_max", and "solution_type"
        
    '''        
    #effective_volumes = np.array([relative_volume(working_catalogue[i]["phot_g_mean_mag"], working_catalogue[i]["parallax"]) for i in range(len(working_catalogue))])
    #effective_volumes = generate_rolling_average(working_catalogue)
        
    # precompute the q-L mappings for all the objects
    if grids is None:
        p_boundaries = np.linspace(*p_range, model_cube.shape[0]+1)
        q_boundaries = np.linspace(*q_range, model_cube.shape[1]+1)
        grids = np.array(compute_grids(catalogue, sc_cube, p_boundaries, q_boundaries, 
                                        scale=scale, verbose=verbose, combine_higher_order=combine_higher_order))
    
    # precompute the model cube for each object, masked and renormalized according to their maximum mass ratio
    masked_model_cubes = mask_and_renormalize_model_cube(model_cube, catalogue, np.linspace(*q_range, model_cube.shape[1]+1))
    
    # compute the likelihood for each binary fraction
    fbs = np.linspace(0.02,0.98,resolution)
    likelihoods = np.zeros(resolution)
    iterator = tqdm(range(resolution), desc="Computing likelihoods") if verbose else range(resolution)
    for i in iterator:
        likelihoods[i] = calculate_log_likelihood(grids, masked_model_cubes, catalogue["solution_type"], fbs[i], cutoff=cutoff)

    return fbs, likelihoods