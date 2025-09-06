import numpy as np
from scipy.spatial import KDTree

class Distance: 

    def move_from_A_to_B_with_x1_displacement(self, A, B, deltas, epsilon=1e-3):
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
    
    def closest_point(self, point, contour):
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

        # Find the index of the contour point closest to the input point
        closest_index = tree.query(point)[1]

        # If the result is an array (e.g., due to batch input), extract the scalar index
        if not isinstance(closest_index, np.int64): 
            closest_index = closest_index[0]

        # Retrieve the actual closest point using the index
        closest_point = contour[closest_index]

        return closest_point

    def closest_border_point(self, border_points, contour): 
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

    def euclidean_distance(self, point1, point2):
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