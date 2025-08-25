import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np

from .common_functions import closest_point, closest_border_point, move_from_A_to_B_with_x1_displacement 
from .common_functions import get_multi_dim_border_points, det_constraints, constraint_bounds, real_world_constraints
from .common_functions import balance_dataset, check_class_balance, convert_columns


def alpha_binary_search(model, point, opp_point, point_target, epsilon=1e-3, max_iter=100):
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
    
    epsilon: float 
        The difference between starting and ending alpha values should be less than epsilon defined in the arguments of the call.

    max_iter : int 
        The number of iterations needed to find an appropriate alpha. This is the limit for the number of iterations for binary search.

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
    """
    start, end = 0, 1  

    for j in range(max_iter): 

        mid = (start + end) / 2 
        mid_point = (1 - mid) * point + mid * opp_point 
        pred = model.predict([mid_point])[0]

        if pred == point_target: 
            start = mid 
        else: 
            end = mid 
        
        if abs(start - end) < epsilon: 
            break 
    
    return (start + end) / 2


def multi_alpha_binary_search(model, points, opp_points, point_target, epsilon=1e-3, max_iter=100): 
    zeros_df, ones_df = np.zeros((points.shape[0], 1)), np.ones((points.shape[0], 1))
    alpha_df = pd.DataFrame(data=np.hstack((zeros_df, ones_df)), columns=['begin', 'end'])
    
    for _ in range(max_iter): 
        # Look for midpoints 
        mids = (alpha_df['begin'] + alpha_df['end']) / 2 
        mids = np.reshape(mids, (-1, 1))
        mid_points = (1 - mids) * points + mids * opp_points
        # Make predictions for the model from mid points
        preds = model.predict(mid_points)
        rows = np.where(preds == point_target)[0]
        other_rows = np.where(preds != point_target)[0]
        # Fix beginning and ending columns 
        true_idx, false_idx = rows, other_rows
        alpha_df.loc[true_idx, 'begin'] = mids[true_idx,0] 
        alpha_df.loc[false_idx, 'end'] = mids[false_idx,0]

        if (np.abs(alpha_df['begin'] - alpha_df['end']) < epsilon).all(): 
            break 
        # For testing 
        # assert true_idx.shape[0] + false_idx.shape[0] == alpha_df.shape[0]
    return (alpha_df['begin'] + alpha_df['end']) / 2 


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
    categorical_features = X.select_dtypes(include='int32').columns.tolist()

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

    # After creating cluster_a and cluster_b
    preds_a = model.predict(cluster_a)
    preds_b = model.predict(cluster_b)

    if isinstance(preds_a, np.ndarray) and len(preds_a.shape) > 1: 
        preds_a = np.argmax(preds_a, axis=1) 
        preds_b = np.argmax(preds_b, axis=1)

    correct_a = cluster_a[preds_a == label_a]
    correct_b = cluster_b[preds_b == label_b]

    num_pairs = min(threshold, correct_a.shape[0] * correct_b.shape[0])  # Avoid overflow
    a_indices = np.random.choice(correct_a.shape[0], num_pairs, replace=True)
    b_indices = np.random.choice(correct_b.shape[0], num_pairs, replace=True)

    boundary_points = []
    for idx in range(num_pairs):
        point = correct_a[a_indices[idx]]
        match_point = correct_b[b_indices[idx]]
        alpha = alpha_binary_search(model, point, match_point, label_a, epsilon=epsilon)
        boundary = (1 - alpha) * point + alpha * match_point
        boundary_points.append(boundary)

    # Convert list to DataFrame with original columns
    boundary_pts = pd.DataFrame(data=boundary_points, columns=X.columns)
    
    # Round categoricals to int for discrete values
    for col in categorical_features: 
        boundary_pts[col] = boundary_pts[col].astype(int)

    return boundary_pts

def multi_decision_boundary(model, X, y, epsilon=1e-3, threshold=10000, batch=10000): 
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

    # After creating cluster_a and cluster_b
    preds_a = model.predict(cluster_a)
    preds_b = model.predict(cluster_b)

    if isinstance(preds_a, np.ndarray) and len(preds_a.shape) > 1: 
        preds_a = np.argmax(preds_a, axis=1) 
        preds_b = np.argmax(preds_b, axis=1)

    correct_a = cluster_a[preds_a == label_a]
    correct_b = cluster_b[preds_b == label_b]

    num_pairs = min(threshold, correct_a.shape[0] * correct_b.shape[0])  # Avoid overflow
    a_indices = np.random.choice(correct_a.shape[0], num_pairs, replace=True)
    b_indices = np.random.choice(correct_b.shape[0], num_pairs, replace=True)

    boundary_points = [] 
    for idx in range(0, num_pairs, batch):
        points = correct_a[a_indices[idx:idx+batch]]
        match_points = correct_b[b_indices[idx:idx+batch]]
        alphas = multi_alpha_binary_search(model, points, match_points, label_a, epsilon=epsilon)
        alphas = np.reshape(alphas, (-1, 1))
        boundary_pts = (1 - alphas) * points + alphas * match_points
        boundary_points.append(boundary_pts)

    # Convert list to DataFrame with original columns
    boundary_pts = np.concatenate(boundary_points, axis=0) 
    boundary_pts = pd.DataFrame(data=boundary_pts, columns=X.columns)
    return boundary_pts

def optimal_point(dataset, model, desired_class, original_class, chosen_row=-1, threshold=10000, point_epsilon=1e-3, epsilon=0.01, constraints=[], delta=15, plot=False): 
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
    
    chosen_row :  int 
        The selected row of the dataset to find the counterfactual explanation for
    
    threshold : int, optional
        Max number of decision boundary points to sample. Default is 10000.
    
    point_epsilon : float, optional
        Precision used to estimate decision boundary points. Default is 0.001 or 1e-3.
    
    epsilon : float, optional
        Step size used when displacing a point past the decision boundary. Default is 0.01.
    
    constraints : list, optional
        A list of real-world constraints on the features (e.g., ranges, logic constraints). Default is [].
    
    delta : int, optional
        Tolerances or maximum displacement for each continuous feature

    plot : boolean 
        Used as a parameter to determine whether to plot the results or not

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

    # Convert categorical columns if needed (before balancing)
    inv_col_map = convert_columns(dataset)

    # Extract features and labels before balancing
    X_orig = dataset.iloc[:, :-1]
    
    # Save the original row's feature values
    undesired_coords = X_orig.iloc[chosen_row, :].copy()

    # Balance the dataset
    dataset = balance_dataset(df=dataset, target=dataset.columns[-1])
    
    if not check_class_balance(dataset, target=dataset.columns[-1]):
        raise RuntimeError("Failed to balance classes for binary classification")
    
    sampled_dataset = dataset.sample(n=min(dataset.shape[0], 20000))

    # Extract new training features/labels after balancing
    X_train = sampled_dataset.iloc[:, :-1]
    y_train = sampled_dataset.iloc[:, -1]
    # Train the model
    print("Fitting model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # -------------------------------
    # STEP 2: Find decision boundary
    # -------------------------------
    print("boundary points started generation...")

    # This step uses binary interpolation to get points close to the decision boundary
    boundary_points = multi_decision_boundary(model, X_train, y_train,
                                             threshold=threshold, epsilon=point_epsilon)
    print("boundary points finished.")
    print(boundary_points.shape)
    # Detect categorical features (assumed as int columns)
    categorical_features = X_train.select_dtypes(include=['int32', 'int64', 'int8']).columns.tolist()

    # Round categoricals to int for discrete values
    for col in categorical_features: 
        boundary_points[col] = boundary_points[col].astype(int)

    # -------------------------------
    # STEP 3: Apply real-world constraints (optional)
    # -------------------------------
    # Reduce boundary points based on external rules (e.g., cost limits, physics constraints)
    contours = real_world_constraints(points=boundary_points,
                                      undesired_coords=undesired_coords,
                                      constraints=constraints)
    undesired_datapt = np.reshape(undesired_coords, (1, -1))  # Reshape undesired point to 2D array

    # if plot:
    #     plt.plot(contours[:,0], contours[:,1], lw=0.5, color='red')  # Commented: Plot contours for visualization

    # -------------------------------
    # STEP 4: Find closest point on constrained boundary
     # -------------------------------
    if contours is not None and desired_class != original_class: 
        print("Finding the closest point from the contour line to the point...")
        contours.reset_index(drop=True, inplace=True)
        optimal_datapt = closest_point(undesired_datapt, contour=contours.to_numpy())
        print("Found the closest point from the contour line to the point.")  # Note: Duplicate print, possibly a typo
        D = optimal_datapt - undesired_datapt  # Compute direction vector
        deltas = D * (1+epsilon)  # Scale by (1 + epsilon) to overshoot
        optimal_datapt = move_from_A_to_B_with_x1_displacement(undesired_datapt, optimal_datapt, deltas=deltas)
    elif desired_class == original_class or contours is None: 
        # If we want to *stay within* the same class (more constrained)
        all_constrained_feats = [var for (var,_) in constraints]
        closest_boundedpt = None 
        vars = set(X_train.columns) - set(all_constrained_feats)
        cont_mutable_vars = [X_train.columns.get_loc(col) for col in vars]
        deltas, len_constr = det_constraints(datapt=undesired_datapt[0], vars=cont_mutable_vars, delta=delta)  # Determine constraints
        
        if len_constr > X_train.shape[1]:
            raise Exception("There cannot be more constraints than features")
        else:
            # All n dimensions are constrained, so generate an exact grid of boundary candidates
            bounded_contour_pts = get_multi_dim_border_points(center=undesired_datapt[0],
                                                              extents=deltas,
                                                              step=0.1)
            # np_bounded_contour = np.array(bounded_contour_pts)  # Convert to NumPy array
            # x_values, y_values = np_bounded_contour[:, 0], np_bounded_contour[:, 1]  # Extract x/y for plotting
            # if plot:
            #     plt.scatter(x_values, y_values, marker='o')  # Plot bounded points
            closest_boundedpt = closest_border_point(bounded_contour_pts, contour=boundary_points)  # Find closest on border
        
        D = closest_boundedpt - undesired_datapt  # Compute direction
        optimal_datapt = move_from_A_to_B_with_x1_displacement(undesired_datapt, closest_boundedpt, deltas=D)  # Move point
    
    # Plot original and optimal points with connecting line
    # if plot:
    #     plt.scatter(undesired_datapt[0][0], undesired_datapt[0][1], c = 'r')  # Plot undesired point
    #     plt.text(undesired_datapt[0][0]+0.002, undesired_datapt[0][1]+0.002, 'NH')  # Label 'NH' (e.g., Non-Healthy)
    #     plt.scatter(optimal_datapt[0][0], optimal_datapt[0][1], c = 'g')  # Plot optimal point (changed to green for distinction)
    #     plt.text(optimal_datapt[0][0]+0.002, optimal_datapt[0][1]+0.002, 'NH')  # Label 'H' (e.g., Healthy; adjusted from duplicate 'NH')
    #     plt.plot([undesired_datapt[0][0], optimal_datapt[0][0]], [undesired_datapt[0][1],optimal_datapt[0][1]], linestyle='--')  # Dashed line between points
    
    categorical_features = [col for col in inv_col_map.keys()]
    final_optimal_datapt = [] 

    for col in X_train.columns:
        if col in categorical_features: 
            idx = int(optimal_datapt[0,X_train.columns.get_loc(col)])
            final_optimal_datapt.append(inv_col_map[col][idx])
        else: 
            final_optimal_datapt.append(optimal_datapt[0,X_train.columns.get_loc(col)])

    return final_optimal_datapt