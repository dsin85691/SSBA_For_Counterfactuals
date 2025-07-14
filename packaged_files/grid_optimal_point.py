import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random 
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree
import numba 

from common_functions import closest_point, closest_border_point, euclidean_distance, move_from_A_to_B_with_x1_displacement 
from common_functions import get_multi_dim_border_points, det_constraints, constraint_bounds, real_world_constraints


@numba.njit
def prediction(Z, grid, epsilon): 
    """
    Identify points near the decision boundary by finding close pairs of grid points
    with different predicted classes and computing their midpoints.

    Parameters
    ----------
    Z : numpy.ndarray
        Array of predicted class labels for each point in the grid (shape: (n_grid_points,)).
    
    grid : numpy.ndarray
        Array of grid points in the feature space (shape: (n_grid_points, n_features)).
    
    epsilon : float
        Maximum distance threshold for considering two points as neighbors. Points
        closer than this with different classes are used to compute boundary midpoints.

    Returns
    -------
    list of numpy.ndarray
        List of midpoint arrays representing approximate boundary points.

    Raises
    ------
    None explicitly, but may raise ValueError if shapes of Z and grid mismatch,
    or if numba compilation fails (if @njit is used without numba installed).

    Notes
    -----
    - This function is JIT-compiled with numba for performance, but can run without it.
      Remove @njit if numba is unavailable.
    - Computational complexity is O(n_grid_points^2), which is inefficient for large grids.
      Suitable only for small grids (e.g., low resolution or few features).
    - Uses Euclidean norm (np.linalg.norm) for distance.
    - Usage: Called internally by boundary computation functions to extract transitions.
    """
    boundary_points = []  # List to collect midpoints
    for i in range(len(grid) - 1):  # Outer loop over grid points
            for j in range(i + 1, len(grid)):  # Inner loop to avoid duplicates/self
                # Check if points are close and have different predictions
                if np.linalg.norm(grid[i] - grid[j]) < epsilon and Z[i] != Z[j]:
                    # Append midpoint as boundary approximation
                    boundary_points.append((grid[i] + grid[j]) / 2)  
    return boundary_points

def compute_decision_boundary_points_all_features(model, X, resolution=100, epsilon=0.01):
    """
    Compute decision boundary points in the high-dimensional feature space by
    generating a dense grid, predicting classes, and finding transitions via midpoints.

    Parameters
    ----------
    model : object
        Trained binary or multi-class classifier with a `predict` method that takes
        an array of input points and returns predictions as an array.
        Example: sklearn.linear_model.LogisticRegression instance.
    
    X : pandas.DataFrame
        Input feature dataset (shape: (n_samples, n_features)). Used to determine
        min/max ranges for numeric features and categories for categorical ones.
    
    resolution : int, optional
        Number of points to sample along each feature axis for the grid. Higher values
        increase density but exponentially increase memory/computation. Default is 100.
    
    epsilon : float, optional
        Distance threshold for detecting class changes between grid points. Should be
        tuned based on feature scales (e.g., small for normalized data). Default is 0.01.

    Returns
    -------
    pandas.DataFrame
        DataFrame of unique approximate boundary points, with the same columns as X.

    Raises
    ------
    ValueError
        If grid shapes mismatch during generation, or if model.predict fails.
    
    MemoryError or OverflowError
        Likely for high resolution or many features due to grid size (resolution ** n_features).

    Notes
    -----
    - Generates a full Cartesian grid over all features, flattened to 1D array.
      For n_features = f, grid size = resolution^f, which is feasible only for small f
      (e.g., f<=3) and low resolution (e.g., <=20). For higher dimensions, consider
      sampling or dimensionality reduction instead.
    - For numeric features: Samples evenly from (min-1, max+1).
    - For categorical features: Maps to integer indices (0 to len(categories)-1).
      Note: The code assumes len(categories) == resolution; otherwise, it may produce
      an incorrect grid size (length mismatch). Consider adjusting resolution to match
      max categories or handling categoricals separately (e.g., one-hot encode beforehand).
    - Predictions are made on the entire grid, then boundary midpoints are found.
    - Unique points are taken to remove duplicates.
    - Usage: Visualize high-D boundaries by projecting (e.g., PCA) or for analysis.
      Best for low-D or with small resolution.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = pd.DataFrame({'feat1': [0, 1, 2], 'feat2': [0, 1, 0]})
    >>> model = LogisticRegression().fit(X, [0, 1, 0])
    >>> boundary = compute_decision_boundary_points_all_features(model, X, resolution=10, epsilon=0.1)
    >>> print(boundary.shape)  # e.g., (number_of_boundary_points, 2)
    (15, 2)
    """
    n_features = X.shape[1]  # Number of features
    # A grid that contains resolution^f samples from the f dimensional space where f is the number of features
    grid = np.zeros((resolution ** n_features, n_features))  # Initialize grid array

    # Generates a grid that contains resolution^f samples based on whether the column contains numeric or categorical values 
    # If the column contains numeric types, then the grid generates a column based on subdividing the numeric columns evenly
    for i in range(n_features):
        # Checks if the column is not a column consisting of categorical values
        # If it is not categorical, then the column must be numeric. 
        if not isinstance(X[X.columns[i]].dtype, pd.CategoricalDtype):
            # Sample evenly spaced points, slightly extended beyond data range
            grid[:, i] = np.tile(np.linspace(X.iloc[:, i].min() - 1, X.iloc[:, i].max() + 1, resolution).repeat(resolution ** (n_features - i - 1)), resolution ** i)
        else:
            # For categorical: Get unique categories and map to integers
            cat_array = X.iloc[:, i].astype('category').cat.categories
            cat_array = np.arange(len(cat_array))  # e.g., [0, 1, 2] for 3 categories
            repeats_per_cat = resolution ** (n_features - i - 1)
            tiles = resolution ** i
            col_values = np.tile(np.repeat(cat_array, repeats_per_cat), tiles)
            grid[:, i] = col_values  # Assign to grid; note: length must match resolution^n_features
            
    # Predict the class for each point in the grid
    Z = model.predict(grid)
    # Find points near the decision boundary
    boundary_points = prediction(Z, grid, epsilon)
 
    return pd.DataFrame(np.unique(boundary_points,axis=0), columns=X.columns)  # Unique points as DataFrame


def optimal_point(dataset, model, desired_class, original_class, undesired_coords, resolution=100, point_epsilon=0.1, epsilon=0.01, constraints=[], deltas=[]): 
    """
    Finds the closest point to the decision boundary from an undesired point using a grid-based
    approximation of the boundary, optionally constrained by real-world conditions. This generates
    a counterfactual explanation by minimizing the distance to the boundary while satisfying class
    change requirements and constraints.

    Parameters
    ----------
    dataset : pd.DataFrame
        Full dataset containing features and a final column with class labels.
    
    model : sklearn-like classifier
        A binary classification model with a `.fit()` and `.predict()` method.
    
    desired_class : int or label
        The target class we want the corrected point to belong to.
    
    original_class : int or label
        The actual class label of the undesired point.
    
    undesired_coords : list or array
        The coordinates of the original ("unhealthy") point.
    
    resolution : int, optional
        Number of points to sample along each feature axis for the grid in boundary computation.
        Higher values improve accuracy but increase computation exponentially. Default is 100.
    
    point_epsilon : float, optional
        Distance threshold for detecting class changes in the grid-based boundary search.
        Default is 0.1.
    
    epsilon : float, optional
        Step size used when displacing a point toward the decision boundary (for overshooting).
        Default is 0.01.
    
    constraints : list, optional
        A list of real-world constraints on the features (e.g., ranges, logic constraints).
        Default is [].
    
    deltas : list, optional
        Tolerances or maximum displacements for each feature. Default is [].

    Returns
    -------
    np.ndarray
        A corrected point (optimal_datapt) that satisfies the class change and real-world constraints.

    Raises
    ------
    Exception
        If the number of constraints exceeds the number of features.

    Notes
    -----
    - This variant uses a grid-based approach (`compute_decision_boundary_points_all_features`)
      to approximate the decision boundary, which is more exhaustive but computationally intensive
      for high dimensions or resolutions. Suitable for low-dimensional spaces (e.g., 2-3 features).
    - Trains the model on the dataset, generates boundary points via grid predictions and midpoint
      detection, applies constraints, and finds the closest optimal point.
    - Assumes binary classification and relies on external functions like `real_world_constraints`,
      `closest_point`, `move_from_A_to_B_with_x1_displacement`, `det_constraints`,
      `get_multi_dim_border_points`, `constraint_bounds`, and `closest_border_point`,
      which must be defined elsewhere.
    - Includes plotting for visualization (e.g., contours, points, lines), requiring matplotlib.
      Plots assume 2D for simplicity (e.g., contours[:,0] and [:,1]).
    - Print statements provide progress tracking.
    - If `desired_class != original_class`, overshoots the boundary slightly for class flip.
      Otherwise, handles bounded constraints differently (full grid or partial filtering).
    - Usage: Generate counterfactuals for explainable AI, optimization, or model interpretation.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.svm import SVC
    >>> dataset = pd.DataFrame({'feat1': [0, 1, 2], 'feat2': [0, 1, 0], 'label': [0, 1, 0]})
    >>> model = SVC(kernel='linear')
    >>> undesired_coords = [2, 0]  # Example point from class 0
    >>> optimal = optimal_point(dataset, model, desired_class=1, original_class=0, undesired_coords=undesired_coords, resolution=20)
    >>> print(optimal)  # e.g., array([[1.5, 0.5]])
    """
    X_train, y_train = dataset.iloc[:, 0:dataset.shape[1]-1], dataset.iloc[:, -1]  # Extract features and labels
    n_features = X_train.shape[1]  # Get number of features

    print("fitting model...")
    model.fit(X_train, y_train)  # Train the model
    print("model finished.")

    print("boundary points started generation...")
    # Use grid-based method to approximate boundary points
    boundary_points = compute_decision_boundary_points_all_features(model, X_train, resolution=resolution, epsilon=point_epsilon)
    print("boundary points finished.")

    # Fitting the boundary points to the constraints provided by the real world
    contours = real_world_constraints(points=boundary_points, undesired_coords=undesired_coords, constraints=constraints)
    contours = contours.to_numpy()  # Convert to NumPy for further processing

    # contours = boundary_points  # (Commented: Alternative to use raw boundary)
    undesired_datapt = np.reshape(np.array(list(undesired_coords)), (1, -1))  # Reshape undesired point to 2D array
    # Find the closest point from the undesired point to the contour line
    print("Finding the closest point from the contour line to the point...")
    optimal_datapt = closest_point(undesired_datapt, contour=contours)
    print("Finding the closest point from the contour line to the point.")  # Note: Duplicate print, possibly a typo
    plt.plot(contours[:,0], contours[:,1], lw=0.5, color='red')  # Plot contours (assumes 2D)

    if desired_class != original_class: 
        D = optimal_datapt - undesired_datapt  # Direction vector to boundary
        deltas = D * (1+epsilon)  # Scale to overshoot slightly
        optimal_datapt = move_from_A_to_B_with_x1_displacement(undesired_datapt, optimal_datapt, deltas=deltas)  # Move point
    else: 
        closest_boundedpt = None
        deltas, len_constr = det_constraints(datapt=undesired_datapt[0], changes=deltas)  # Determine constraints (note: param 'changes' may be a typo for 'deltas')
        bounded_contour_pts = None

        if len_constr > n_features: 
            raise Exception("There cannot be more constraints than features")
        elif len_constr == n_features:
            # Generate border points for fully constrained case
            bounded_contour_pts = get_multi_dim_border_points(center=undesired_datapt[0], extents=deltas, step=0.05)
            np_bounded_contour = np.array(bounded_contour_pts)  # To NumPy
            x_values, y_values = np_bounded_contour[:,0], np_bounded_contour[:, 1]  # Extract for plotting (assumes 2D)
            plt.scatter(x_values, y_values, marker='o')  # Plot bounded points
            closest_boundedpt = closest_border_point(bounded_contour_pts, contour=contours)  # Find closest on border (constraints in all dimensions)
        else: 
            # Generate bounded contour points for partial constraints 
            bounded_contour_pts = constraint_bounds(contours, undesired_datapt, deltas)
            closest_boundedpt = closest_point(point=undesired_datapt, contour=bounded_contour_pts)  # Find closest point based on partial constraints

        D = closest_boundedpt - undesired_datapt  # Direction vector
        optimal_datapt = move_from_A_to_B_with_x1_displacement(undesired_datapt, closest_boundedpt, deltas=D)  # Move point
    plt.scatter(undesired_datapt[0][0], undesired_datapt[0][1], c = 'r')  # Plot undesired point
    plt.text(undesired_datapt[0][0]+0.002, undesired_datapt[0][1]+0.002, 'NH')  # Label 'NH' (e.g., Non-Healthy)
    plt.scatter(optimal_datapt[0][0], optimal_datapt[0][1], c = 'r')  # Plot optimal point
    plt.text(optimal_datapt[0][0]+0.002, optimal_datapt[0][1]+0.002, 'NH')  # Label 'NH' (note: duplicate label, perhaps typo for 'H')
    plt.plot([undesired_datapt[0][0], optimal_datapt[0][0]], [undesired_datapt[0][1],optimal_datapt[0][1]], linestyle='--')  # Dashed line between points
    return optimal_datapt