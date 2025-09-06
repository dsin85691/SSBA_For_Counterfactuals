import numpy as np 
import matplotlib.pyplot as plt

class Constraints: 


    def __init__(self, constraints, delta_constraints): 
        self.constraints = constraints 
        self.delta_constraints = delta_constraints
        self.len_constr = len(self.constraints) 

    def get_constraints(self): 
        return self.constraints 

    def get_len_constr(self): 
        return self.len_constr

    def real_world_constraints(self, points, undesired_coords): 
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
        if len(self.constraints) == 0: 
            return points 
        
        for constraint in self.constraints: 
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
    
    def constraint_bounds(self, contours, datapt): 
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
        if len(self.constraints) == 0: 
            raise Exception("No constraints were assigned.")
        bounded_contour = contours.copy()  # Copy to avoid modifying original
        for i in range(len(self.constraints)): 
            if self.constraints[i] > 0:  # Active if >0
                x = datapt[0][i]  # Reference value for dimension i
                # This should just be the delta
                delta_x = self.constraints[i]
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
    
    def det_constraints(self, datapt, vars, deltas): 
        """
        Determine the effective constraints based on deltas, scaling them relative to the data point.

        Parameters
        ----------
        datapt : list or numpy.ndarray
            The data point (feature vector) to scale constraints against.
        
        deltas : list
            A list of percentages in terms of each datapoint's feature value.

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
                if deltas[i] is None: 
                    constraints[i] = 0 
                else:
                    constraints[i] = (deltas[i]/100)*datapt[i]  # Scale as percentage of datapt
                    len_constr+=1
        self.constraints = constraints 
        self.len_constr = len_constr


    def get_multi_dim_border_points(self, center, extents, step=0.1):
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
        extents = self.delta_constraints  # Convert extents to NumPy array
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