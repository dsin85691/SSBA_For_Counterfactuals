from optimal_point import OptimalPoint 
import pandas as pd
import numpy as np
from constraints import Constraints 


class OptimalPointGPU(OptimalPoint): 
        
    def __init__(self, dataset, model): 
        super.__init__(dataset, model)

    def alpha_binary_search(self, points, opp_points, point_target, epsilon=1e-3, max_iter=100):
        zeros_df, ones_df = np.zeros((points.shape[0], 1)), np.ones((points.shape[0], 1))
        alpha_df = pd.DataFrame(data=np.hstack((zeros_df, ones_df)), columns=['begin', 'end'])

        for _ in range(max_iter): 
            # Look for midpoints 
            mids = (alpha_df['begin'] + alpha_df['end']) / 2 
            mids = np.reshape(mids, (-1, 1))
            mid_points = (1 - mids) * points + mids * opp_points
            # Make predictions for the model from mid points
            preds = self.model.predict(mid_points)
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
        self.undesired_datapt = np.reshape(undesired_coords, (1, -1))  # Reshape undesired point to 2D array

        # if plot:
        #     plt.plot(contours[:,0], contours[:,1], lw=0.5, color='red')  # Commented: Plot contours for visualization

        # -------------------------------
        # STEP 4: Find closest point on constrained boundary
        # -------------------------------
        if contours is not None and desired_class != original_class: 
            optimal_datapt = self.unconstrained(
                undesired_datapt=self.undesired_datapt, 
                contours=contours.to_numpy(), 
                epsilon=epsilon
            )
        elif desired_class == original_class or contours is None: 
            optimal_datapt = self.constrained(
                x_train=x_train, 
                contours=contours, 
                undesired_datapt=self.undesired_datapt
            )
        # Plot original and optimal points with connecting line
        if plot:
            self.data_transform.plot()

        self.optimal_datapt = optimal_datapt
        return self.data_transform.return_correct_datapt(datapoint=optimal_datapt)