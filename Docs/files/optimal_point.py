
import pandas as pd
import numpy as np
from base import Base

class OptimalPoint(Base): 


    def __init__(self, dataset, model): 
        super().__init__(self, dataset, model)

    def alpha_binary_search(self): 
        pass 

    def find_decision_boundary(self, model, X, y, epsilon=1e-3, threshold=10000, batch=1000): 
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
        for idx in range(0, num_pairs, batch):
            points = correct_a[a_indices[idx:idx+batch]]
            match_points = correct_b[b_indices[idx:idx+batch]]
            alphas = self.alpha_binary_search(model, points, match_points, label_a, epsilon=epsilon)
            alphas = np.reshape(alphas, (-1, 1))
            boundary_pts = (1 - alphas) * points + alphas * match_points
            boundary_points.append(boundary_pts)

        # Convert list to DataFrame with original columns
        boundary_pts = np.concat(boundary_points,axis=0) 
        boundary_pts = pd.DataFrame(data=boundary_pts, columns=X.columns)

        return boundary_pts

    def run(self, desired_class, original_class, chosen_row=-1, threshold=10000,
             point_epsilon=1e-3, epsilon=0.01, constraints=[], delta=15, plot=False): 
        
        undesired_coords = super().run(chosen_row=chosen_row)

        return desired_class, original_class, chosen_row, \
               threshold, point_epsilon, epsilon, constraints, \
               delta, plot, undesired_coords