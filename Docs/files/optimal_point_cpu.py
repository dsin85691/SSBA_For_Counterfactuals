from optimal_point import OptimalPoint 
import numpy as np
from constraints import Constraints 


class OptimalPointCPU(OptimalPoint): 
        
    def __init__(self, dataset, model): 
        super.__init__(dataset, model)
    
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
            
            if j % 4 == 0:
                if abs(start - end) < epsilon: 
                    break 
        
        return (start + end) / 2


    def run(self, desired_class, original_class, chosen_row=-1, threshold=10000, point_epsilon=1e-3, epsilon=0.01, constraints=[], deltas=[], plot=False): 

        desired_class, original_class, threshold,
        point_epsilon, epsilon, constraints,
        deltas, plot, undesired_coords = super().run(
            desired_class, original_class, 
            chosen_row, threshold,
            point_epsilon, epsilon, 
            constraints, deltas, plot
        )
        
        # -------------------------------
        # STEP 2: Find decision boundary
        # -------------------------------
        print("boundary points started generation...")
        x_train, y_train = self.data_transform.separate()

        # This step uses binary interpolation to get points close to the decision boundary
        boundary_points = self.find_decision_boundary(self.model, x_train, y_train, 
                                                      threshold=threshold, epsilon=point_epsilon)
        print("boundary points finished.")
        print(boundary_points.shape)
        # -------------------------------
        # STEP 3: Apply real-world constraints (optional)
        # -------------------------------

        self.constraints = Constraints(constraints=constraints, 
                                       delta_constraints=deltas)

        # Reduce boundary points based on external rules (e.g., cost limits, physics constraints)
        contours = self.constraints.real_world_constraints(points=boundary_points,
                                        undesired_coords=undesired_coords,
                                        constraints=constraints)
        undesired_datapt = np.reshape(undesired_coords, (1, -1))  # Reshape undesired point to 2D array

        # if plot:
        #     plt.plot(contours[:,0], contours[:,1], lw=0.5, color='red')  # Commented: Plot contours for visualization

        # -------------------------------
        # STEP 4: Find closest point on constrained boundary
        # -------------------------------
        if contours is not None and desired_class != original_class: 
            optimal_datapt = self.unconstrained(
                undesired_datapt=undesired_datapt, 
                contours=contours.to_numpy(), 
                epsilon=epsilon
            )
        elif desired_class == original_class or contours is None: 
            optimal_datapt = self.constrained(
                x_train=x_train, 
                contours=contours, 
                undesired_datapt=undesired_datapt
            )
        # Plot original and optimal points with connecting line
        if plot:
            self.data_transform.plot()

        self.optimal_datapt = optimal_datapt
        return self.data_transform.return_correct_datapt(datapoint=optimal_datapt)