# Segmented Sampling for Boundary Approximation (SSBA) for Counterfactual Explanations

## Methodology ##
Counterfactual explanations are the backbone of our methodology, and they help us formulate how we approach the generation of counterfactuals alongside the ability to compare similar lines of prior research. As defined by Dandl and Molnar in their book "Interpretable Machine Learning," a counterfactual explanation can be defined as "the smallest change to the feature values that changes the prediction to a predefined output" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). Please refer to the book or similar references for further explanation of the definition of a counterfactual explanation.

We describe the methodology in the paper. In summary, the method computes the nearest counterfactual explanation (unconstrained) by way of a method of generating discrete decision boundary points. We generate such decision boundary points with the segmented sampling boundary approximation (SSBA) method, which involves taking a binary search approach for two distinct classes and minimizing the distance between pairs of points from differing classes. Once we have generated a set of discrete boundary points, we then choose the boundary point that is minimally distant from the original instance. We add a small $\epsilon$ value such that the prediction $f(x)$ on that original instance $x$ flips in its predicted class value.

## Documentation Notebook Explained ## 

Within our ``documentation_notebook.ipynb `` notebook file, we provide the details of our SSBA method alongside an implementation of the grid-based approach. The grid-based approach is based on the generation of a discrete grid of points $G \subset \mathbf{R}^f$, and then we search across the entire grid to find pairs of points that differ in class value. Our SSBA method is implemented within the functions ```find_decision_boundary``` and ```alpha_binary_search.``` Our main function for generating counterfactual explanations and constrained/bounded counterfactuals that we describe in the paper is called ```optimal_point()``` in the Jupyter notebook. The ```optimal_point()``` function computes a given counterfactual explanation/bounded counterfactual satisfying a given set of criteria. Given a labelled observational dataset of binary classes, we can apply the method to take a given point and produce the counterfactual explanation for that point. Within the function, we make use of the ```find_decision_boundary``` and ```alpha_binary_search``` functions to compute the decision boundary points needed for the generation of counterfactual explanations. From this list of points, we then apply the Euclidean metric with KDTrees to find the closest point from our set of decision boundary points to the original instance/query point. 

Note: There are two different $\epsilon$ parameters that each control a separate functionality. The first $\epsilon$, called $\epsilon$ in the code, is used to ensure that the generated counterfactual has a different predicted class value from the original instance. Given that we choose a decision boundary point $d$ for the counterfactual explanation, we add a small $\epsilon$ such that $f(d + \epsilon) \neq f(x)$ where $x$ is our original point. The second $\epsilon$ is called ```point_epsilon``` in the code. This parameter is used to ensure that the distance between pairs of points from differing classes is smaller than $\epsilon$, which we generally assign to be $1.0 * 10^{-3}$ as the default. Using a small $\epsilon$ value ensures that the method can generate points that are very close to the boundary. 

## Additional Details ##

This repository contains all of the notebooks used to demonstrate the method, as well as packaging the relevant files under the ```docs/files``` directory. 

As for our Jupyter Notebooks, we have one documentation notebook ```documentation_notebook.ipynb``` that documents the functions and their relevant parameters, return types, and values. Under our ```docs/tests `` folder, we have several notebooks that not only test the method, but also have a notebook where we compare the computational costs of the grid-based approach with our binary search method as described in the paper. 

# Code Summary # 

Summary of Notebooks under ```docs/tests```: 
1. ```dice_ml_comparison_gpu.ipynb``` is the notebook that we used for the comparison of our method against DiCE's model-agnostic approaches as shown in the paper.
2. ```alibi_comparison.ipynb``` is the notebook that we used for the comparison of our method against Alibi's gradient approach.
3. ```computation_testing_notebook_gpu.ipynb``` is the notebook that we used for the comparison of our method with the grid-based approach in terms of time complexity. We provide time complexity results for both CPU/GPU.
4. ```binary_search_tests.ipynb``` is a notebook that we used for testing the generation of decision boundary points for a logistic regression model. Since we know that boundary points must have equal probability between both classes for logistic regression models, we check that  the model maintains $50\%$ probabilities between each class even as we change the generated synthetic dataset or change the size of the feature space. 
5. ```ablation_study_point_epsilon.ipynb```, ```ablation_study_increasing_n.ipynb```, ```ablation_study_resolution.ipynb``` are three files that we record results for our ablation studies. We study the effect of changing three different parameters used in our methodologies. One is the resolution size $R$ for the grid-based approach, which is the size for a given dimension of the grid of size $R^f \subset \textbf{R}^f$ (Think of a square in two dimensions or a cube in three dimensions). The second parameter we study is the parameter $\epsilon$, which is the distance between pairs of points from different classes. As shown in the Jupyter notebook, decreasing $\epsilon$ yields points that are approximately close to boundary points. The third parameter is $n$ which is the number of features or the size of the feature space. In the ```ablation_study_increasing_n.ipynb```, we check the robustness of our methodology across multiple different values of $n$ ranging from $2$ to $10,000$. 

All original files used at the beginning of the research project are shown under the ```docs/original_files``` folder. 

# Getting Started # 

Here is an easy way to get started. Once you have set up your GPU environment through RAPIDS, we recommend running this code for testing purposes. Please also refer to ```ablation_study_increasing_n.ipynb``` or ```binary_search_tests.ipynb``` for getting started.


First, let's import all of the required libraries and tools for running the code.
```
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import warnings
import random

random.seed(0)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

# This imports the decision boundary function generator
# you can also try: from docs.files.binary_search_optimal_point import multi_decision_boundary
from files.binary_search_optimal_point import multi_decision_boundary 
```

Seccond, let's load our GPU environment. Please refer to the docs for the RAPIDS GPU environment. 
```
# Runs the RAPIDS GPU environment
%load_ext cuml.accel
```

Third, let's preprocess our dataset and construct our model. 
```
# Generates a synthetic dataset
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, random_state=42)
# creates a logistic regression classifier
model = LogisticRegression()
y = y.reshape(-1,1)
df1 = pd.DataFrame(data=np.hstack((X,y)))
x_train = df1.iloc[:,:-1]
y_train = df1.iloc[:,-1]
```

Fourth, let's fit our model to the dataset. 
```
# Fits the dataset to the model
model.fit(x_train,y_train)
```

Finally, let us generate the boundary points for the given dataset. 
```
# Generates the boundary points and outputs the boundary points as a numpy array
boundary_points = multi_decision_boundary(model, x_train, y_train, threshold=100000, epsilon=1e-4)

# Print the decision boundary points and the shape for the boundary points
# This should be of the form (N, p) where $N$ is the number of data points and $p$ is the number of features
print("Decision Boundary Points (All Features):")
print(boundary_points)
print(boundary_points.shape)
```



# How to Set up Environment and Run Files # 

1. Set up a environment using conda, Linux virtual environments, Google Colab environment, etc.
2. Run Jupyter Notebook within VSCode, Google Colab.
3. Make sure to run the appropriate environments for those files. We recommend only running the GPU environment for the files with the label gpu in the name such as ```documentation_notebook_gpu.ipynb``` or ```dice_ml_comparison_gpu.ipynb.``` We can also run the GPU environment for the ablation studies. 


# Setting up the GPU Environment # 

Constructing the GPU environment can be done at the link: https://docs.rapids.ai/install/?_gl=1*1km83yv*_ga*MTcxOTEzMjc1NC4xNzU1MTA2Njc1*_ga_RKXFW6CM42*czE3NTYwODA3NjgkbzckZzEkdDE3NTYwODExNDgkajYwJGwwJGgw. To set up RAPIDS, ensure you have CUDA installed or an Nvidia GPU preinstalled on your computer before starting the installation. Make sure to choose "specific packages," and choose cuDF, cuML, and JupyterLab for comparable results. 

So, a command like this (lets you set up with conda): 

```conda create -n gpu_env -c rapidsai -c conda-forge -c nvidia cudf=25.08 cuml=25.08 python=3.12 'cuda-version>=12.0,<=12.9'  jupyterlab```

Once you have completed this, you should run the ```gpu_requirements.txt``` that includes the other files needed. This file contains the notebooks that are not Alibi-specific (one notebook). 

# Setting up Alibi-specific environment # 

To run the Alibi Jupyter Notebook, you need to run an environment specific to Alibi, which we provide. Create a new environment and run the ```alibi_requirements.txt``` file. This will let you run the alibi comparison notebook under ```Docs/tests``` folder.

# Setting up the CPU environment # 

Make sure to run the ```cpu_environment.txt``` file after generating your environment. 


# Errors # 

If you encounter an SSL certificate error, ensure your Linux environment, WSL environment, etc., is updated to the latest version of Ubuntu or Debian to facilitate the successful installation and execution of `dice_ml` files. 

You may also submit an issue request for updates/changes. 

Also, if you want to fork a change and add it to the repository, I will review your request and see if I can approve of the changes. 

# BibTeX / Reference # 

If you ever use our work in a corresponding paper, research work, or application, we would highly encourage you to reference us in your work.  

```
@misc{sin2025geometrical,
  author       = {Sin, Daniel and Toutounchian, Milad},
  title        = {Towards Personalized Treatment Plan: Geometrical Model-Agnostic Approach to Counterfactual Explanations},
  year         = {2025},
  howpublished = {Manuscript submitted to arXiv},
  note         = {Preprint},
}
```

# Link to Paper # 

Here is a link to the now available publication, "Towards Personalized Treatment Plan: Geometrical Model-Agnostic Approach to Counterfactual Explanations." Please check it out, and we are welcome to hear feedback back from the community. 

Link: https://arxiv.org/abs/2510.22911

# Analogues / Additional References # 

Our algorithm is technically a discrete analogue of the bisection method applied for machine learning classification problems. 

Reference:
https://en.wikipedia.org/wiki/Bisection_method
https://en.wikipedia.org/wiki/Nested_intervals

# Points of Discussion # 

There are a few exceptional cases where the algorithm may not work based on research. Pathological examples like the random classifier might make it impossible to find a solution. We do require that there will be a finite number of class changes within a line segment. Majority of machine learning models will not behave like the pathological example. 

In this case, a random classifier would be some model that selects each class with equal probability. For a random classifier, every point is a boundary point (50% probability for each class), so there is no convergence in this case. 

# Feedback # 

We love any recommendations from the community. If you have ever further questions, we would love to discuss with you on our work post release on arxiv. Please write a issues request, and we can discuss further on any potential comments or rooms for improvement.

# Acknowledgement # 

Professor Milad Toutounchian (faculty at Drexel) supported my work throughout this project, spanning the year from November 2024 to present (November 2025). His support has helped the initiation and progression of this project. 





