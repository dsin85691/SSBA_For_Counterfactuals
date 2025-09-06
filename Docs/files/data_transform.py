
from sklearn.utils import resample
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class DataTransform: 

    def __init__(self, dataset): 
        self.dataset = dataset 
        self.target = self.dataset.columns[-1]
        self.inv_col_map = None 
        self.x_train = dataset.iloc[:, :-1]
        self.y_train = dataset.iloc[:, -1]
        self.categorical_features = self.x_train.select_dtypes(include=['int32',
                                                                         'int64', 'int8']).columns.tolist() 
        self.continuous_features = [col for col in self.dataset.columns 
                                    if col not in self.cat_features]
    
    def get_dataset(self): 
        return self.dataset

    def separate(self): 
        return self.x_train, self.y_train
    
    def get_inverse_map(self): 
        return self.inv_col_map
    
    def convert_columns(self): 
        inv_col_map = {} 
        df = self.dataset

        for col in df.columns: 
            any_string = df[col].apply(lambda x: isinstance(x, str)).any()

            if not any_string: 
                df[col] = df[col].astype('float64')
            else: 
                inv_col_map[col] = {}
                unique_vals = np.unique(df[col])

                for i, val in enumerate(unique_vals): 
                    inv_col_map[col][i] = val 
                    df.loc[df[col] == val, col] = i

                df[col] = df[col].astype('int32')

        self.inv_col_map = inv_col_map

    def balance_dataset(self): 
        df = self.dataset
        target = self.target 

        unique_vals = df[target].unique()
        max_samples = df[target].value_counts().max()

        balanced_dataset = pd.DataFrame(data=[], columns=df.columns)

        for val in unique_vals: 
            class_subset = df[df[target] == val]
            if class_subset.shape[0] != max_samples: 
                upsampled_class = resample(class_subset, 
                                        replace=True, 
                                        n_samples=max_samples, 
                                        random_state=42)  # for reproducibility
            else: 
                upsampled_class = class_subset
            balanced_dataset = pd.concat([balanced_dataset, upsampled_class], ignore_index=True)

        balanced_dataset[target] = balanced_dataset[target].astype('int32')
        self.dataset = balanced_dataset

    def sample(self, num_samples): 
        min_samples = min(self.dataset.shape[0], num_samples) 
        sampled_dataset = self.dataset.sample(n=min_samples)
        return sampled_dataset 
    
    def check_class_balance(self): 
        df = self.dataset 
        target = self.target
        class_counts = df[target].value_counts()
        print("Class counts:\n", class_counts)
        
        if class_counts.nunique() == 1:
            return True 
        else:
            return False
    
    def plot(self, contours, undesired_datapt, optimal_datapt): 
        params = {'mathtext.default': 'regular' }
        plt.rcParams.update(params)
        plt.scatter(contours[:,0], contours[:,1], lw=0.5, color='purple', label="Decision Boundary Points")  # Commented: Plot contours for visualization
        plt.scatter(undesired_datapt[0][0], undesired_datapt[0][1], c = 'r', label="NH: Not Healthy")  # Plot undesired point
        plt.text(undesired_datapt[0][0]+0.002, undesired_datapt[0][1]+0.002, 'NH')  # Label 'NH' (e.g., Non-Healthy)
        plt.scatter(optimal_datapt[0][0], optimal_datapt[0][1], c = 'g', label="H: Healthy")  # Plot optimal point (changed to green for distinction)
        plt.text(optimal_datapt[0][0]+0.002, optimal_datapt[0][1]+0.002, 'NH')  # Label 'H' (e.g., Healthy; adjusted from duplicate 'NH')
        plt.plot([undesired_datapt[0][0], optimal_datapt[0][0]], [undesired_datapt[0][1],optimal_datapt[0][1]], linestyle='--')  # Dashed line between points
        red_patch = mpatches.Patch(color='red', label='Not Healthy')
        blue_patch = mpatches.Patch(color='blue', label='Healthy')
        green_patch = mpatches.Patch(color='green', label="Counterfactual")
        purple_patch = mpatches.Patch(color='purple', label='Decision Boundary Point')
        plt.legend(loc='lower left', handles=[red_patch, blue_patch, purple_patch, green_patch])
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title("Toy Dataset")
        plt.show()

    def return_correct_datapt(self, datapoint):     
        final_datapoint = [] 
        x_train, _ = self.separate()
        inv_col_map = self.get_inverse_map() 

        for col in x_train:
            if col in self.categorical_features: 
                idx = int(datapoint[0,x_train.columns.get_loc(col)])
                final_datapoint.append(inv_col_map[col][idx])
            else: 
                final_datapoint.append(datapoint[0,x_train.columns.get_loc(col)]) 
        
        return final_datapoint
    
    