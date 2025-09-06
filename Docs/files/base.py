
from data_transform import DataTransform
from euclidean_distance import Distance 
import numpy as np
import matplotlib.pyplot as plt

class Base:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model  
        self.data_transform = DataTransform(dataset=self.dataset)
        self.distance_tool = Distance()
        self.constraints = None 
        self.undesired_datapt = None 
        self.optimal_datapt = None

    def get_undesired_datapt(self): 
        return self.undesired_datapt 

    def get_optimal_datapt(self): 
        return self.optimal_datapt

    def find_decision_boundary(self): 
        pass 

    def unconstrained(self, undesired_datapt, contours, epsilon): 
        print("Finding the closest point from the contour line to the point...")
        contours.reset_index(drop=True, inplace=True)
        optimal_datapt = self.distance_tool.closest_point(undesired_datapt, contour=contours)
        print("Found the closest point from the contour line to the point.")  # Note: Duplicate print, possibly a typo
        D = optimal_datapt - undesired_datapt  # Compute direction vector
        deltas = D * (1+epsilon)  # Scale by (1 + epsilon) to overshoot
        self.optimal_datapt = self.distance_tool.move_from_A_to_B_with_x1_displacement(undesired_datapt, optimal_datapt, deltas=deltas)
        return self.optimal_datapt 
    
    def constrained(self, x_train, contours, undesired_datapt, plot): 
        # If we want to *stay within* the same class (more constrained)
        all_constrained_feats = [var for (var,_) in self.constraints]
        closest_boundedpt = None 
        vars = set(x_train.columns) - set(all_constrained_feats)
        cont_mutable_vars = [x_train.columns.get_loc(col) for col in vars]
        deltas, len_constr = self.constraints.det_constraints(datapt=undesired_datapt[0], vars=cont_mutable_vars, delta=delta)  # Determine constraints
        
        if len_constr > x_train.shape[1]:
            raise Exception("There cannot be more constraints than features")
        elif len_constr < x_train.shape[1]: 
            # Partially constrained - less than n dimensions are constrained
            bounded_contour_pts = self.constraints.constraint_bounds(contours, undesired_datapt, deltas)  # Apply partial bounds
            closest_boundedpt = self.distance_tool.closest_point(point=undesired_datapt, contour=bounded_contour_pts)  # Find closest
        else:
            # All n dimensions are constrained, so generate an exact grid of boundary candidates
            bounded_contour_pts = self.constraints.get_multi_dim_border_points(center=undesired_datapt[0],
                                                            extents=deltas,
                                                            step=0.1)
            np_bounded_contour = np.array(bounded_contour_pts)  # Convert to NumPy array
            if plot:
                x_values, y_values = np_bounded_contour[:, 0], np_bounded_contour[:, 1]  # Extract x/y for plotting
                plt.scatter(x_values, y_values, marker='o')  # Plot bounded points
            closest_boundedpt =  self.distance_tool.closest_border_point(bounded_contour_pts, contour=contours)  # Find closest on border
        
        D = closest_boundedpt - undesired_datapt  # Compute direction
        self.optimal_datapt = self.distance_tool.move_from_A_to_B_with_x1_displacement(undesired_datapt, closest_boundedpt, deltas=D)  # Move point
        return self.optimal_datapt
    
    def run(self, chosen_row):
        x_train, y_train = self.data_transform.separate()

        # Convert categorical columns if needed (before balancing)
        self.data_transform.convert_columns()

        # Save the original row's feature values
        undesired_coords = x_train.iloc[chosen_row, :].copy()

        # Balance the dataset
        self.data_transform.balance_dataset()
        
        if not self.data_transform.check_class_balance():
            raise RuntimeError("Failed to balance classes for binary classification")
        
        sampled_dataset = self.data_transform.sample(n=20000)

        # Extract new training features/labels after balancing
        X_train = sampled_dataset.iloc[:, :-1]
        y_train = sampled_dataset.iloc[:, -1]

        # Train the model
        print("Fitting model...")
        self.model.fit(X_train, y_train)
        print("Model training complete.")

        self.data_transform = DataTransform(dataset=sampled_dataset)

        return undesired_coords
    
    def get_boundary_points(self): 
        return self.boundary_points