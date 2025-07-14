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


def alpha_binary_search(model, point, opp_point, point_target, opp_target, epsilon=0.01):
    """
    Perform a binary search along the line segment between two points to find the
    approximate alpha value where the model's prediction changes from one target
    label to another. This is useful for approximating decision boundaries in
    binary classification by finding the transition point along a segment connecting
    points from opposite classes.

    Parameters
    ----------
    model : object
        A trained machine learning model with a `predict` method that takes a list
        or array of input points and returns predictions as an array. The model
        should be a binary classifier (e.g., from scikit-learn, PyTorch, etc.).
        Example: sklearn.linear_model.LogisticRegression instance.
    
    point : numpy.ndarray
        A 1D array representing the starting point (feature vector) in the feature
        space, typically from one class. Must have the same shape as `opp_point`.
    
    opp_point : numpy.ndarray
        A 1D array representing the opposing point (feature vector) in the feature
        space, typically from the opposite class. Must have the same shape as `point`.
    
    point_target : int or str
        The expected prediction label for the `point`. This is used to initialize
        the search and compare against the model's prediction at interpolated points.
        Should match the model's output format (e.g., 0 or 1 for binary classes).
    
    opp_target : int or str
        The expected prediction label for the `opp_point`. This should be different
        from `point_target` and is used to detect when the prediction flips.
    
    epsilon : float, optional
        The tolerance for convergence in the binary search. The loop stops when the
        difference between the search bounds is less than this value. Default is 0.01.
        Smaller values yield more precise alphas but may increase computation time.

    Returns
    -------
    float
        The approximate alpha value (between 0 and 1) where the model's prediction
        transitions. A value closer to 0 means the boundary is nearer to `point`,
        while closer to 1 means nearer to `opp_point`.

    Raises
    ------
    None explicitly, but may raise exceptions from `model.predict` if the input
    shapes are incompatible or if the model is not properly trained.

    Notes
    -----
    - This function assumes the decision boundary is crossed exactly once along the
      line segment; multiple crossings (e.g., in non-linear models) may lead to
      approximate or incorrect results.
    - The binary search updates bounds based on prediction matches, but if no flip
      occurs (e.g., both points predicted the same), it will converge to a midpoint
      without a true boundary.
    - Usage: Typically called within a loop over pairs of points from different
      classes to sample multiple boundary points.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression().fit(np.array([[0], [1]]), [0, 1])
    >>> point = np.array([0.0])
    >>> opp_point = np.array([1.0])
    >>> alpha = alpha_binary_search(model, point, opp_point, 0, 1, epsilon=0.001)
    >>> print(alpha)  # Approximately 0.5 for a linear boundary at 0.5
    0.5
    """
    start, end = 0, 1  # Initialize search bounds: 0 at 'point', 1 at 'opp_point'
    while abs(end - start) >= epsilon:  # Loop until convergence within epsilon
        alpha = (start + end) / 2  # Midpoint alpha (float division ensured)
        # Interpolate: weighted average between point and opp_point
        temp_candidate = (1 - alpha) * point + alpha * opp_point
        # Predict on the interpolated point (assumes model.predict returns array)
        temp_target = model.predict([temp_candidate])[0]
        if temp_target == point_target: 
            start = alpha  # Move start bound if prediction matches point's target
            point_target = temp_target  # Update target (though often redundant)
        elif temp_target == opp_target: 
            end = alpha  # Move end bound if prediction matches opp's target
            opp_target = temp_target  # Update target (though often redundant)
    return (start + end) / 2  # Return midpoint as approximate transition alpha


def find_decision_boundary(model, X, y, epsilon=1e-3, threshold=10000):
    """
    Approximate the decision boundary of a binary classification model by sampling
    points along line segments between correctly classified points from opposite
    classes. Uses binary search to find transition points and collects them into
    a DataFrame. Handles categorical features by rounding them to integers.

    Parameters
    ----------
    model : object
        A trained binary classification model with a `predict` method that takes
        a list or array of input points and returns predictions as an array.
        Example: sklearn.svm.SVC instance.
    
    X : pandas.DataFrame
        The feature dataset, where rows are samples and columns are features.
        Supports mixed types, including integer categoricals.
    
    y : pandas.Series or numpy.ndarray
        The target labels corresponding to X. Must contain exactly two unique
        binary labels (e.g., 0 and 1).
    
    epsilon : float, optional
        The precision for the binary search in `alpha_binary_search`. Smaller
        values increase accuracy but computation time. Default is 1e-3.
    
    threshold : int, optional
        The maximum number of boundary points to generate. Stops early if reached
        to prevent excessive computation on large datasets. Default is 10000.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the approximated boundary points, with the same
        columns as X. Categorical columns (detected as int types) are converted
        to integers.

    Raises
    ------
    ValueError
        If y does not contain exactly two unique labels (non-binary classification).

    Notes
    -----
    - This function clusters points by true labels (y), then filters pairs where
      the model correctly predicts them (to ensure opposite sides of the boundary).
    - It may miss boundaries if the model has high error rates (few correct pairs).
    - Computational complexity is O(n*m) where n and m are cluster sizes, capped
      by threshold. For large datasets, reduce threshold or sample clusters.
    - A `bool_vec` is created but unused; it may be a remnant for future masking
      (e.g., to ignore categoricals in interpolation).
    - Categorical features are auto-detected as int columns and rounded to int
      in the output for interpretability.
    - Usage: Call after training a model to visualize or analyze its boundary,
      e.g., plot the points in 2D or use for explanations.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.svm import SVC
    >>> X = pd.DataFrame({'feat1': [0, 1, 2], 'feat2': [0, 1, 0]})
    >>> y = np.array([0, 1, 0])
    >>> model = SVC(kernel='linear').fit(X, y)
    >>> boundary = find_decision_boundary(model, X, y, epsilon=0.001, threshold=5)
    >>> print(boundary.shape)  # e.g., (number_of_points, 2)
    (2, 2)
    """
    # Detect categorical features (assumed as int columns)
    categorical_features = X.select_dtypes(include=int).columns.tolist()
    cat_indices = [X.columns.get_loc(col) for col in categorical_features]

    # Create a boolean vector (1 for continuous, 0 for categorical; currently unused)
    bool_vec = [1] * (len(X.columns)) 
    for i in range(len(cat_indices)): 
        bool_vec[cat_indices[i]] = 0 

    X_np = X.to_numpy()  # Convert features to NumPy for efficient ops
    y_np = y.to_numpy() if not isinstance(y, np.ndarray) else y  # Ensure y is NumPy
    boundary_points = []  # List to collect boundary point arrays
    unique_labels = np.unique(y_np)  # Get unique class labels
    if len(unique_labels) != 2:
        raise ValueError("Only supports binary classification.")
    
    label_a, label_b = unique_labels[0], unique_labels[1]  # Assign labels

    # Cluster points by true labels
    cluster_a = X_np[y_np == label_a]
    cluster_b = X_np[y_np == label_b]

    total_N = 0  # Counter for generated points
    for i in range(cluster_a.shape[0]):
        point = cluster_a[i]
        pt_pred = model.predict([point])  # Predict on point from cluster A

        for j in range(cluster_b.shape[0]): 
            match_point = cluster_b[j]
            match_pt_pred = model.predict([match_point])  # Predict on point from B
            # Check if model correctly classifies both (ensures opposite sides)
            if pt_pred.item() == label_a and match_pt_pred.item() == label_b: 
                # Find alpha where prediction flips
                alpha = alpha_binary_search(model, point, match_point, label_a, label_b, epsilon=epsilon)
                # Compute boundary point via interpolation
                boundary = (1 - alpha) * point + alpha * match_point
                boundary_points.append(boundary)

                total_N += 1
                if total_N >= threshold:  # Early stop inner loop
                    break
        if total_N >= threshold:  # Early stop outer loop
            break
    
    # Convert list to DataFrame with original columns
    boundary_pts = pd.DataFrame(data=boundary_points, columns=X.columns)

    # Round categoricals to int for discrete values
    for col in categorical_features: 
        boundary_pts[col] = boundary_pts[col].astype(int)

    return boundary_pts


def optimal_point(dataset, model, desired_class, original_class, undesired_coords, threshold=10000, point_epsilon=0.1, epsilon=0.01, constraints=[], deltas=[]): 
    """
    Finds the closest point to the decision boundary from an undesired point,
    optionally constrained by real-world conditions.
    This essentially finds the counterfactual explanation for a given point by minimizing the distance to the given boundary.
    This method is important because it addresses a key problem with the original optimal_point() function where we generated an R^n dimensional grid that we would then have to iterate over. 
    The problem with iterating over such a grid is eventually that we will hit a memory error for high-dimensional features such as 20, 30 or 40 features. This will cause the function to crash. 
    Additionally, due to the exponential increase of the number of features to search, the grid will become infeasible to search (curse of dimensionality). 

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
    
    threshold : int, optional
        Max number of decision boundary points to sample. Default is 10000.
    
    point_epsilon : float, optional
        Precision used to estimate decision boundary points. Default is 0.1.
    
    epsilon : float, optional
        Step size used when displacing a point toward the decision boundary. Default is 0.01.
    
    constraints : list, optional
        A list of real-world constraints on the features (e.g., ranges, logic constraints). Default is [].
    
    deltas : list, optional
        Tolerances or maximum displacements for each feature. Default is [].

    Returns
    -------
    np.ndarray
        A corrected point that satisfies the class change and real-world constraints.

    Raises
    ------
    Exception
        If the number of constraints exceeds the number of features.

    Notes
    -----
    - This function trains the model on the provided dataset, generates boundary points using
      `find_decision_boundary`, applies constraints, and finds the closest optimal point.
    - Assumes binary classification and relies on external functions like `real_world_constraints`,
      `closest_point`, `move_from_A_to_B_with_x1_displacement`, etc., which must be defined elsewhere.
    - Includes plotting for visualization (e.g., boundary contours, points), which requires matplotlib.
    - The function blends boundary approximation with counterfactual generation, useful for explainable AI.
    - Print statements are for progress tracking; plotting is partially commented out but can be enabled.
    - Usage: Call with a dataset and model to generate counterfactuals, e.g., for model interpretation or optimization.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> dataset = pd.DataFrame({'feat1': [0, 1, 2], 'feat2': [0, 1, 0], 'label': [0, 1, 0]})
    >>> model = LogisticRegression()
    >>> undesired_coords = [2, 0]  # Example point from class 0
    >>> optimal = optimal_point(dataset, model, desired_class=1, original_class=0, undesired_coords=undesired_coords)
    >>> print(optimal)  # e.g., array([[1.5, 0.5]])
    """
    # -------------------------------
    # STEP 1: Train the model
    # -------------------------------
    X_train = dataset.iloc[:, 0:-1]  # Extract features from dataset
    y_train = dataset.iloc[:, -1]  # Extract labels from dataset
    n_features = X_train.shape[1]  # Get number of features

    print("fitting model...")
    model.fit(X_train, y_train)  # Train the model on the dataset
    print("model finished.")

    # -------------------------------
    # STEP 2: Find decision boundary
    # -------------------------------
    print("boundary points started generation...")

    # This step uses binary interpolation to get points close to the decision boundary
    boundary_points = find_decision_boundary(model, X_train, y_train,
                                             threshold=threshold, epsilon=point_epsilon)
    print("boundary points finished.")
    print(boundary_points.shape)

    # -------------------------------
    # STEP 3: Apply real-world constraints (optional)
    # -------------------------------
    # Reduce boundary points based on external rules (e.g., cost limits, physics constraints)
    contours = real_world_constraints(points=boundary_points,
                                      undesired_coords=undesired_coords,
                                      constraints=constraints)
    contours = np.unique(contours.to_numpy(), axis=0)  # Remove duplicates from constrained points
    undesired_datapt = np.reshape(np.array(list(undesired_coords)), (1, -1))  # Reshape undesired point to 2D array

    # -------------------------------
    # STEP 4: Find closest point on constrained boundary
    # -------------------------------
    print("Finding the closest point from the contour line to the point...")
    optimal_datapt = closest_point(undesired_datapt, contour=contours)
    print("Finding the closest point from the contour line to the point.")  # Note: Duplicate print, possibly a typo
    #plt.plot(contours[:,0], contours[:,1], lw=0.5, color='red')  # Commented: Plot contours for visualization


    # -------------------------------
    # STEP 5: Post-process based on class flip requirement
    # -------------------------------

    # If we want to *flip* the class of the point...
    if desired_class != original_class: 
         # Move in the direction of the boundary, slightly overshooting
        D = optimal_datapt - undesired_datapt  # Compute direction vector
        deltas = D * (1+epsilon)  # Scale by (1 + epsilon) to overshoot
        optimal_datapt = move_from_A_to_B_with_x1_displacement(undesired_datapt, optimal_datapt, deltas=deltas)
    else:
        # If we want to *stay within* the same class (more constrained)
        closest_boundedpt = None
        deltas, len_constr = det_constraints(datapt=undesired_datapt[0], deltas=deltas)  # Determine constraints

        if len_constr > X_train.shape[1]:
            raise Exception("There cannot be more constraints than features")

        elif len_constr == X_train.shape[1]:
            # All n dimensions are constrained, so generate an exact grid of boundary candidates
            bounded_contour_pts = get_multi_dim_border_points(center=undesired_datapt[0],
                                                              extents=deltas,
                                                              step=0.05)
            np_bounded_contour = np.array(bounded_contour_pts)  # Convert to NumPy array
            x_values, y_values = np_bounded_contour[:, 0], np_bounded_contour[:, 1]  # Extract x/y for plotting
            plt.scatter(x_values, y_values, color='blue', marker='o')  # Plot bounded points
            closest_boundedpt = closest_border_point(bounded_contour_pts, contour=contours)  # Find closest on border

        else:
            # Partially constrained - less than n dimensions are constrained
            bounded_contour_pts = constraint_bounds(contours, undesired_datapt, deltas)  # Apply partial bounds
            closest_boundedpt = closest_point(point=undesired_datapt, contour=bounded_contour_pts)  # Find closest
        
        D = closest_boundedpt - undesired_datapt  # Compute direction
        optimal_datapt = move_from_A_to_B_with_x1_displacement(undesired_datapt, closest_boundedpt, deltas=D)  # Move point
    
    # Plot original and optimal points with connecting line
    plt.scatter(undesired_datapt[0][0], undesired_datapt[0][1], c = 'r')  # Plot undesired point
    plt.text(undesired_datapt[0][0]+0.002, undesired_datapt[0][1]+0.002, 'NH')  # Label 'NH' (e.g., Non-Healthy)
    plt.scatter(optimal_datapt[0][0], optimal_datapt[0][1], c = 'g')  # Plot optimal point (changed to green for distinction)
    plt.text(optimal_datapt[0][0]+0.002, optimal_datapt[0][1]+0.002, 'H')  # Label 'H' (e.g., Healthy; adjusted from duplicate 'NH')
    plt.plot([undesired_datapt[0][0], optimal_datapt[0][0]], [undesired_datapt[0][1],optimal_datapt[0][1]], linestyle='--')  # Dashed line between points
    return optimal_datapt