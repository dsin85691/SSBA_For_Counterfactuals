import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
from scipy.spatial import KDTree
from sklearn.utils import resample
import cupy as cp
import cudf
random.seed(0)
warnings.filterwarnings('ignore', category=UserWarning)


def closest_point(point, contour):
    """
    Finds the closest point on a contour to a given reference point.

    Parameters:
    -----------
    point : array-like or tuple
        A single point (e.g., [x, y]) for which the nearest contour point is to be found.

    contour : array-like of shape (n_points, n_dimensions)
        A list or array of points representing the contour. Each point should have the same dimensionality as `point`.

    Returns:
    --------
    closest_point : array-like
        The point on the contour that is closest to the input `point`.
    """

    # Build a KD-tree for fast nearest neighbor search over the contour points
    tree = KDTree(contour)

    # Query nearest neighbor (k=1)
    dist, ind = tree.query([point], k=1)

    # Extract scalar index
    closest_index = int(ind[0][0])

    # Get closest point
    closest_point = contour[closest_index]

    return closest_point

def closest_border_point(border_points, contour):
    """
    Finds the point in `border_points` that is closest to any point in the given `contour`.

    Parameters:
    -----------
    border_points : array-like of shape (n_points, n_dimensions)
        A list or array of candidate points (e.g., border or edge points).

    contour : array-like of shape (m_points, n_dimensions)
        A list or array of contour points to which the closest distance is measured.

    Returns:
    --------
    min_point : array-like
        The point from `border_points` that is closest to any point in the `contour`.
    """

    # Build a KDTree for efficient nearest neighbor queries on contour points
    tree = KDTree(contour)

    # Initialize variables to track the closest border point and the smallest distance found
    min_point = None         # Will hold the closest point from `border_points`
    total_min = float('inf') # Initialize the minimum distance as infinity

    # Iterate through each candidate border point
    for border_point in border_points:
        # Query the KDTree to find the distance to the closest contour point
        dist, _ = tree.query(border_point)

        # If this distance is the smallest encountered so far, update tracking variables
        if dist < total_min:
            total_min = dist
            min_point = border_point

    # Return the border point with the minimum distance to the contour
    return min_point

def euclidean_distance(point1, point2):
    """
    Computes the Euclidean distance between two points.

    Parameters:
    -----------
    point1 : array-like
        The first point (e.g., [x1, y1] or [x1, y1, z1]).

    point2 : array-like
        The second point (e.g., [x2, y2] or [x2, y2, z2]).

    Returns:
    --------
    float
        The Euclidean distance between `point1` and `point2`.
    """

    # Convert both points to NumPy arrays, subtract them element-wise,
    # and compute the L2 norm (i.e., Euclidean distance) of the result
    return np.linalg.norm(np.array(point1) - np.array(point2))


def move_from_A_to_B_with_x1_displacement(A, B, deltas, epsilon=1e-3):
    """
    Move from point A to point B in n-dimensional space with a desired movement in the x1 dimension.

    Parameters:
    - A: list or np.array, coordinates of the starting point A
    - B: list or np.array, coordinates of the target point B
    - delta_x1: float, the desired displacement in the x1 dimension

    Returns:
    - P: np.array, coordinates of the new point after moving delta_x1 along x1-axis
    """
    A = np.array(A)
    B = np.array(B)

    # Calculate direction vector from A to B
    D = B - A

    # Calculate the scaling factor t for the desired movement in x1
    t = deltas / (D + epsilon)   # D[0] is the x1 component of the direction vector

    # Calculate the new point P based on t
    P = A + t * D

    print(t)
    print(D)

    return P


def get_multi_dim_border_points(center, extents, step=0.1):
    """
    Generate points on the boundaries of an n-dimensional hyperrectangle.

    Parameters
    ----------
    center : list or numpy.ndarray
        The center of the hyperrectangle, a list or array of length n (number of dimensions).

    extents : list or numpy.ndarray
        The full widths (diameters) in each dimension, a list or array of length n.
        Note: The code uses half-widths internally (extents / 2).

    step : float, optional
        Step size for sampling points along each dimension's grid. Smaller values
        increase density but computation time. Default is 0.1.

    Returns
    -------
    list of tuples
        Each tuple represents a point on the boundary of the hyperrectangle.

    Raises
    ------
    None explicitly, but may raise ValueError if center and extents have mismatched lengths,
    or TypeError if inputs are not array-like.

    Notes
    -----
    - Uses a set to avoid duplicate points, which can occur at corners/edges.
    - For each dimension, fixes the boundary (min/max) and grids over others.
    - Handles 1D case specially.
    - Suitable for generating boundary samples in constrained optimization or
      visualization of feasible regions in n-D space.
    - Output as list of tuples for easy conversion to arrays if needed.

    Examples
    --------
    >>> center = [0, 0]
    >>> extents = [2, 2]  # Rectangle from (-1,-1) to (1,1)
    >>> points = get_multi_dim_border_points(center, extents, step=0.5)
    >>> print(len(points))  # e.g., number of sampled boundary points
    16
    """
    center = np.array(center)  # Convert center to NumPy array
    extents = np.array(extents)  # Convert extents to NumPy array
    n = len(center)  # Number of dimensions
    points = set()   # Use set to avoid duplicates

    # Define min and max bounds for each dimension (using half-extents)
    bounds = [(c - e / 2, c + e / 2) for c, e in zip(center, extents)]

    # For each dimension, generate points on the lower and upper boundaries
    for dim in range(n):
        # For lower and upper boundary in this dimension
        for bound_val in [bounds[dim][0], bounds[dim][1]]:
            # Generate grid points for all other dimensions
            other_dims = [i for i in range(n) if i != dim]
            ranges = [np.arange(bounds[i][0], bounds[i][1] + step, step) for i in other_dims]
            if not ranges:  # Handle 1D case
                points.add(tuple([bound_val] if dim == 0 else []))
                continue
            # Create meshgrid for other dimensions
            grids = np.meshgrid(*ranges, indexing='ij')
            coords = [grid.ravel() for grid in grids]

            # Construct points
            for coord in zip(*coords):
                point = [0] * n
                # Set the current dimension to the boundary value
                point[dim] = bound_val
                # Set other dimensions to the grid values
                for i, val in zip(other_dims, coord):
                    point[i] = val
                points.add(tuple(point))  # Add as tuple to set

    return list(points)

def det_constraints(datapt, vars, delta):
    """
    Determine the effective constraints based on deltas, scaling them relative to the data point.

    Parameters
    ----------
    datapt : list or numpy.ndarray
        The data point (feature vector) to scale constraints against.

    deltas : int
        A percentage in terms of the datapoint feature value. If float/int, it's treated as a percentage
        (e.g., 10 means 10% of datapt[i]); otherwise ignored.

    Returns
    -------
    tuple
        (constraints: list of scaled delta values or -1 if not applicable,
         len_constr: int count of active constraints)

    Raises
    ------
    None explicitly, but may raise TypeError if datapt/deltas are incompatible.

    Notes
    -----
    - Initializes constraints as [-1] * len(deltas), updating only for numeric deltas.
    - Scaling: constraint[i] = (deltas[i] / 100) * datapt[i], assuming percentages.
    - Used to count and quantify constraints for bounded regions.
    - Usage: Pre-process deltas before applying bounds in optimization or counterfactuals.

    Examples
    --------
    >>> datapt = [100, 200]
    >>> deltas = [10, 'none']  # 10% for first, ignore second
    >>> constraints, len_constr = det_constraints(datapt, deltas)
    >>> print(constraints, len_constr)  # [10.0, -1], 1
    [10.0, -1] 1
    """
    constraints = [-1] * datapt.shape[0]  # Initialize with -1 (inactive)
    len_constr = 0  # Counter for active constraints
    for i in range(len(datapt)):
        if i in vars:  # Check if numeric
            constraints[i] = (delta/100)*datapt[i]  # Scale as percentage of datapt
            len_constr+=1  # Increment counter
        else:
            constraints[i] = 0
            len_constr+=1
    return constraints, len_constr

def constraint_bounds(contours, datapt, constraints):
    """
    Filter contour points to those within specified bounds based on constraints.

    Parameters
    ----------
    contours : numpy.ndarray
        Array of contour points (shape: (n_points, n_features)).

    datapt : numpy.ndarray
        The reference data point (shape: (1, n_features)) to center bounds around.

    constraints : list
        List of delta values (bounds widths) for each feature; >0 activates filtering.

    Returns
    -------
    numpy.ndarray
        Filtered contour points within the bounds.

    Raises
    ------
    Exception
        If no constraints are assigned (all <=0).

    Notes
    -----
    - For each active constraint, computes [x - delta/2, x + delta/2] and filters.
    - Sequentially applies filters, potentially reducing points cumulatively.
    - Includes plotting of bounds (vertical/horizontal lines for dims 0/1).
    - Assumes 2D for plotting; extend for higher dims if needed.
    - Usage: Constrain boundary points in optimization, e.g., feasible counterfactuals.

    Examples
    --------
    >>> contours = np.array([[0,0], [1,1], [2,2], [3,3]])
    >>> datapt = np.array([[1,1]])
    >>> constraints = [2, 2]  # Bounds width 2 for each
    >>> bounded = constraint_bounds(contours, datapt, constraints)
    >>> print(bounded)  # e.g., array([[0,0], [1,1], [2,2]])
    """
    if len(constraints) == 0:
        raise Exception("No constraints were assigned.")
    bounded_contour = contours.copy()  # Copy to avoid modifying original
    for i in range(len(constraints)):
        if constraints[i] > 0:  # Active if >0
            x = datapt[0][i]  # Reference value for dimension i
            # This should just be the delta
            delta_x = constraints[i]
            # Generate a lower and upper bounds on each constraint
            lowb_x, highb_x = x - (delta_x / 2), x + (delta_x / 2)
            contour_arr = bounded_contour[:, i]  # Extract column i
            # Choose the correct indices for the multi-dimensional df
            indices = np.where((contour_arr >= lowb_x) & (contour_arr <= highb_x))
            bounded_contour_pts = bounded_contour[indices]  # Filter rows
            bounded_contour = bounded_contour_pts  # Update
            if i == 0:  # Plot vertical lines for dim 0
                plt.axvline(x=highb_x, color='b', linestyle='-', label='High Bound x')
                plt.axvline(x=lowb_x, color='b', linestyle='-', label='Low bound x')
            else:   # Plot horizontal for dim 1 (assumes 2D)
                plt.axhline(y=highb_x, color='b', linestyle='-', label='High Bound y')
                plt.axhline(y=lowb_x, color='b', linestyle='-', label='Low bound y')
    return bounded_contour

def real_world_constraints(points, undesired_coords, constraints):
    """
    Filter points based on real-world constraints relative to undesired_coords.

    Parameters
    ----------
    points : pandas.DataFrame
        DataFrame of points to filter, with feature columns.

    undesired_coords : list or array
        The reference ("undesired") point coordinates, matching points' features.

    constraints : list of lists, optional
        Each sublist: [feature_name (str), operator ('equal', 'greater', or other for <)].
        Empty list returns points unchanged.

    Returns
    -------
    pandas.DataFrame
        Filtered points satisfying all constraints.

    Raises
    ------
    None explicitly, but may raise KeyError if feature_name not in points.columns,
    or IndexError if constraints malformed.

    Notes
    -----
    - Sequentially applies filters: == for 'equal', > for 'greater', < otherwise.
    - Uses column index from get_loc for comparison value.
    - Useful for imposing domain-specific rules, e.g., in counterfactual explanations.
    - If constraints empty, returns original points.
    - Usage: Post-process boundary points to respect real-world feasibility.

    Examples
    --------
    >>> points = pd.DataFrame({'pt1': [1,2,3], 'pt2': [4,5,6], 'pt3': [10,11,12]}, columns=['feat1','feat2','feat3'])
    >>> undesired_coords = [1,2,5]
    >>> constraints = [['feat1', 'greater']]
    >>> filtered = real_world_constraints(points, undesired_coords, constraints)
    >>> print(filtered)  # pd.DataFrame(data=[[4,5,6], [10,11,12]]) # Since feat1 = 1 for the undesired point, we select all points that have feat1 > 1
    >>> constraints = [['feat1', 'greater'], ['feat2', 'less']].
    >>> filtered = real_world_constraints(points, undesired_coords, constraints)
    >>> print(filtered)  # pd.DataFrame(data=[]) # Since feat2 = 2 for the undesired point, we select all points that have feat2 < 2.
    """
    if len(constraints) == 0:
        return points

    for constraint in constraints:
        select_pts = None
        if constraint[1] == "equal":  # Filter equal to undesired value
            select_pts = points.loc[points[constraint[0]] == undesired_coords[points.columns.get_loc(constraint[0])], :]
        elif constraint[1] == "greater":  # Filter greater than
            select_pts = points.loc[points[constraint[0]] > undesired_coords[points.columns.get_loc(constraint[0])], :]
        else:  # Default: less than
            select_pts = points.loc[points[constraint[0]] < undesired_coords[points.columns.get_loc(constraint[0])], :]

        points = select_pts  # Update with filtered
        if points.shape[0] == 0:
            return None
    return points

def convert_columns(df):
    inv_col_map = {}

    for col in df.columns:
        any_string = df[col].apply(lambda x: isinstance(x, str)).any()

        if not any_string:
            df[col] = df[col].astype('float64')
        else:
            inv_col_map[col] = {}
            unique_vals = np.unique(df[col])

            for i, val in enumerate(unique_vals):
                inv_col_map[col][i] = val
                df.loc[df[col] == val, col] = i

            df[col] = df[col].astype('int32')

    return inv_col_map

def balance_dataset(df, target):

    unique_vals = df[target].unique()
    max_samples = df[target].value_counts().max()

    balanced_dataset = pd.DataFrame(data=[], columns=df.columns)

    for val in unique_vals:
        class_subset = df[df[target] == val]
        if class_subset.shape[0] != max_samples:
            upsampled_class = resample(class_subset,
                                    replace=True,
                                    n_samples=max_samples,
                                    random_state=42)  # for reproducibility
        else:
            upsampled_class = class_subset
        balanced_dataset = pd.concat([balanced_dataset, upsampled_class], ignore_index=True)

    balanced_dataset[target] = balanced_dataset[target].astype('int32')
    return balanced_dataset

def check_class_balance(df, target):
    class_counts = df[target].value_counts()
    print("Class counts:\n", class_counts)

    if class_counts.nunique() == 1:
        return True
    else:
        return False

def gaussian_noise_generator(mean_vector, cov_matrix, size):
   return np.random.multivariate_normal(mean_vector, cov_matrix, size)

def gaussian_sample_generator(sample, classifier, dataset, cov_matrix, columns, label, size):
    gaussian_noise_samples = gaussian_noise_generator(mean_vector=sample, cov_matrix=cov_matrix/25, size=size)
    gaussian_preds = classifier.predict(gaussian_noise_samples)
    indices = np.where(gaussian_preds == label)
    gaussian_noise_samples = gaussian_noise_samples[indices]
    gaussian_preds = gaussian_preds[indices]
    dataset = pd.concat([dataset, pd.DataFrame(data=gaussian_noise_samples, columns=columns)], ignore_index=True)
    return dataset # Return dataset



def gaussian_noise_generator_gpu(mean_matrix, cov_matrix, size):
    """
    Generate Gaussian samples on GPU for multiple mean vectors at once.

    Parameters
    ----------
    mean_matrix : array-like, shape (B, d) or (d,)
        Mean vectors. If 1D, treated as a single mean vector.
    cov_matrix : array-like, shape (d, d)
        Covariance matrix.
    size : int
        Number of samples to generate per mean vector.

    Returns
    -------
    samples : cp.ndarray, shape (B*size, d)
        Generated Gaussian samples.
    """
    mean_matrix = cp.asarray(mean_matrix)
    cov_matrix = cp.asarray(cov_matrix)

    # Handle single mean vector case
    if mean_matrix.ndim == 1:
        mean_matrix = mean_matrix[None, :]  # shape (1, d)

    B, d = mean_matrix.shape

    # Cholesky decomposition
    L = cp.linalg.cholesky(cov_matrix)  # (d, d)

    # Generate noise for all batches at once
    z = cp.random.standard_normal((B, size, d))  # (B, size, d)

    # Apply covariance
    transformed = z @ L.T  # (B, size, d)

    # Add means with broadcasting
    samples = transformed + mean_matrix[:, None, :]  # (B, size, d)

    # Reshape to (B*size, d) for classifier compatibility
    return samples.reshape(B * size, d)

def gaussian_sample_generator_gpu(samples, classifier, dataset, cov_matrix, columns, labels, size):
    # --- Generate all samples in batch ---
    gaussian_noise_samples = gaussian_noise_generator_gpu(
        mean_matrix=samples,  # (B, d)
        cov_matrix=cov_matrix / 25,
        size=size
    )  # (B*size, d)

    # Expand labels so each sample has a label
    labels = np.repeat(labels, size)  # shape (B*size,)

    # --- Move to CPU for sklearn classifier ---
    gaussian_noise_samples_cpu = cp.asnumpy(gaussian_noise_samples)
    gaussian_preds = classifier.predict(gaussian_noise_samples_cpu)

    # --- Filtering back on GPU ---
    gaussian_preds_gpu = cp.asarray(gaussian_preds)
    mask = gaussian_preds_gpu == cp.asarray(labels)
    gaussian_noise_samples = gaussian_noise_samples[mask]
    dataset_gpu = cp.asarray(dataset)

    # --- Back to pandas ---
    dataset_gpu = cp.concatenate([dataset_gpu, gaussian_noise_samples], axis=0)

    return cp.asnumpy(dataset_gpu)