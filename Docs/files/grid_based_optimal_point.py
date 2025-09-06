from base import Base 
import pandas as pd 
import numpy as np 
import numba 

class GridBased(Base): 

    def __init__(self, dataset, model): 
        super().__init__(self, dataset, model)

    def find_decision_boundary(self, resolution=100, epsilon=0.01): 
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
        X, _ = self.data_transform.separate()
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
        Z = self.model.predict(grid)
        # Find points near the decision boundary
        boundary_points = self.prediction(Z, grid, epsilon)

        if len(boundary_points) == 0: 
            raise Exception("No Boundary Points Found. The DataFrame is empty. Please try to change the parameters.")
    
        return pd.DataFrame(np.unique(boundary_points,axis=0), columns=X.columns)  # Unique points as DataFrame
    

    @numba.njit
    def prediction(self, Z, grid, epsilon): 
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