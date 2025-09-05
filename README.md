# Segmented Sampling for Boundary Approximation (SSBA) for Counterfactual Explanations



Counterfactual explanations are the backbone of our methodology, and they help us formulate how we approach the generation of counterfactuals alongside the ability to compare similar lines of prior research. As defined by Dandl and Molnar in their book "Interpretable Machine Learning," a counterfactual explanation can be defined as "the smallest change to the feature values that changes the prediction to a predefined output" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). Based on their definition, we define the following criteria for a counterfactual explanation. The first criterion is that the "prediction of the counterfactual must match the predefined outcome exactly" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). We would like to have "minimal changes in the features" so that the predicted probability switches to the desired class. Another criterion is that "a counterfactual should be as similar as possible to the instance regarding feature values" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). The counterfactual should be close to the original instance and should also "change as few features as possible" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). The final criterion is that "a counterfactual instance should have feature values that are likely" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). To simplify, the produced counterfactual should make logical sense. Our dataset consists of an observational dataset of persons, where each person has a set of mutable and immutable features. Some of these mutable features might be hair length or nail length. Age might be an immutable feature depending on the circumstance. If the counterfactual produced is of the same person with a lower age, we know that this is incorrect or factually not possible. While there are exceptions, height generally does not decrease, so a counterfactual producing a younger and shorter person would be factually incorrect. We should maintain the generation of a counterfactual explanation and set constraints such that the produced counterfactual explanation would align with these constraints (Mahajan et. al., [2019](https://arxiv.org/abs/1912.03277)).

We describe the methodology in the paper, but to recapitulate, the method computes the nearest counterfactual explanation (unconstrained) by way of a method of generating discrete decision boundary points. We generate such decision boundary points with the segmented sampling boundary approximation (SSBA) method, which involves taking a binary search approach for two distinct classes and minimizing the distance between pairs of points from differing classes. Once we have generated a set of discrete boundary points, we then choose the boundary point that is minimally distant from the original instance. We add a small $ \epsilon`$ value such that the prediction $f(x)$ on that original instance $x$ flips in its predicted class value.

Within our ``documentation_notebook.ipynb `` notebook file, we provide the details of our SSBA method alongside an implementation of the grid-based approach. The grid-based approach is based on the generation of a discrete grid of points $G \subset \mathbf{R}^f$, and then we search across the entire grid to find pairs of points that differ in class value. Our SSBA method is implemented within the functions ```find_decision_boundary``` and ```alpha_binary_search.``` Our main function for generating counterfactual explanations and constrained/bounded counterfactuals that we describe in the paper is called ```optimal_point()``` in the Jupyter notebook. The ```optimal_point()``` function computes a given counterfactual explanation/bounded counterfactual satisfying a given set of criteria. Given a labelled observational dataset of binary classes, we can apply the method to take a given point and produce the counterfactual explanation for that point. Within the function, we make use of the ```find_decision_boundary``` and ```alpha_binary_search``` functions to compute the decision boundary points needed for the generation of counterfactual explanations. From this list of points, we then apply the Euclidean metric with KDTrees to find the closest point from our set of decision boundary points to the original instance/query point. 

Note: There are two different $\epsilon$ parameters that each control a separate functionality. The first $\epsilon$, called $\epsilon$ in the code, is used to ensure that the generated counterfactual has a different predicted class value from the original instance. Given that we choose a decision boundary point $d$ for the counterfactual explanation, we add a small $\epsilon$ such that $f(d + \epsilon) \neq f(x)$ where $x$ is our original point. The second $\epsilon$ is called ```point_epsilon``` in the code. This parameter is used to ensure that the distance between pairs of points from differing classes is smaller than $\epsilon$, which we generally assign to be $1.0 * 10^{-3}$ as the default. Using a small $\epsilon$ value ensures that the method can generate points that are very close to the boundary. 

This repository contains all of the notebooks used to demonstrate the method, as well as packaging the relevant files under the ```docs/files``` directory. 

As for our Jupyter Notebooks, we have one documentation notebook ```documentation_notebook.ipynb``` that documents the functions and their relevant parameters, return types, and values. Under our ```docs/tests `` folder, we have several notebooks that not only test the method, but also have a notebook where we compare the computational costs of the grid-based approach with our binary search method as described in the paper. 

Recapitulation of Notebooks under ```docs/tests```: 
1. ```dice_ml_comparison_gpu.ipynb``` is the notebook that we used for the comparison of our method against DiCE's model-agnostic approaches as shown in the paper.
2. ```alibi_comparison.ipynb``` is the notebook that we used for the comparison of our method against Alibi's gradient approach.
3. ```computation_testing_notebook_gpu.ipynb``` is the notebook that we used for the comparison of our method with the grid-based approach in terms of time complexity. We provide time complexity results for both CPU/GPU.
4. ```binary_search_tests.ipynb``` is a notebook that we used for testing the generation of decision boundary points for a logistic regression model. Since we know that boundary points must have equal probability between both classes for logistic regression models, we check that the $50\%-50\%$ probabilities between each class are maintained for varying numbers of features for a given synthetic dataset.
5. ```ablation_study_point_epsilon.ipynb``` and ```ablation_study_resolution.ipynb``` are two files that we record results for our ablation studies. We study the effect of changing two parameters. One is the resolution size $R$, which is the size for a given dimension of the grid of size $R^f \subset \em{R}^f$ (Think of a square in two dimensions or a cube in three dimensions). The second parameter we study is the parameter $\epsilon$, which is the distance between pairs of points from different classes. As shown in the Jupyter notebook, decreasing $\epsilon$ yields points that are approximately close to boundary points.

All original files used at the beginning of the research project are shown under the ```docs/original_files``` folder. 

Link to the Paper: https://drive.google.com/file/d/14_MB8i8rU60r-kF9WZfVOtF3g0PHd1n-/view?usp=drive_link

# How to Set up Environment and Run Files # 

1. Set up a GPU using conda, Linux virtual environments, Google Colab environment, etc.
2. Run Jupyter Notebook within VSCode, Google Colab.
3. Make sure to run the appropriate environments for those files.


# Setting up the GPU Environment # 

Constructing the GPU environment can be done at the link: https://docs.rapids.ai/install/?_gl=1*1km83yv*_ga*MTcxOTEzMjc1NC4xNzU1MTA2Njc1*_ga_RKXFW6CM42*czE3NTYwODA3NjgkbzckZzEkdDE3NTYwODExNDgkajYwJGwwJGgw. To set up RAPIDS, ensure you have CUDA installed or an Nvidia GPU preinstalled on your computer before starting the installation. Make sure to choose "specific packages," and choose cuDF, cuML, and JupyterLab for comparable results. 

So, a command like this (lets you set up with conda): 

```conda create -n gpu_env -c rapidsai -c conda-forge -c nvidia cudf=25.08 cuml=25.08 python=3.12 'cuda-version>=12.0,<=12.9'  jupyterlab```

Once you have completed this, you should run the ```gpu_requirements.txt``` that includes the other files needed. This file contains the notebooks that are not Alibi-specific (one notebook). 

# Setting up Alibi-specific environment # 

To run the AlibiAlibi Jupyter Notebook, you need to run an environment specific to Alibi, which we provide. Create a new environment and run the ```alibi_requirements.txt``` file. This will let you run the alibi comparison notebook under ```Docs/tests``` folder.





# Errors # 

If you encounter an SSL certificate error, ensure your Linux environment, WSL environment, etc., is updated to the latest version of Ubuntu or Debian to facilitate the successful installation and execution of `dice_ml` files.

You may also submit an issue request for updates/changes.


# Acknowledgement # 

Professor Milad Toutouchian supported my work throughout this project, spanning 10-11 months from November 2024 to September 2025. His support has helped the initiation and progression of this project. 





