# Optimal Point for Counterfactual Explanations

As defined by Dandl and Molnar in their book "Interpretable Machine Learning," a counterfactual explanation can be defined as "the smallest change to the feature values that changes the prediction to a predefined output" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). 

Based on their definition, we define the following criteria for a counterfactual explanation. The first criterion is that the "prediction of the counterfactual must match the predefined outcome exactly" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)).We would like to have "minimal changes in the features" so that the predicted probability switches to the desired class. 

Another criterion is that "a counterfactual should be as similar as possible to the instance regarding feature values" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). The counterfactual should be close to the original instance and should also "change as few features as possible" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). 

The final criterion is that "a counterfactual instance should have feature values that are likely" (Dandl & Molnar, [2019](https://christophm.github.io/interpretable-ml-book/)). To simplify, the produced counterfactual should make logical sense. Let's say that our dataset consists of a observational dataset of persons where each person has a set of mutable and immutable features. Some of these mutable features might be hair length or nail length. Age might be an immutable feature depending on the circumstance. If the counterfactual produced is of the same person with a lower age, we know that this is incorrect or factually not possible. While there are exceptions, height generally does not decrease, so a counterfactual producing a younger and shorter person would be factually incorrect. We should maintain in the generation of a counterfactual explanation and set constraints such that the produced counterfactual explanation would align with these constraints (Mahajan et. al., [2019](https://arxiv.org/abs/1912.03277)).


We describe the methodology in the paper below. The repository consists of a model-agnostic function ```optimal_point()``` which computes the counterfactual explanation satisfying the above criterions. Given a labelled observational dataset of binary classes, we can apply the method to take a given point and produce the counterfactual explanation for that point. Through our method, we find points on the decision boundary and construct a list of points that lie on the contour. From this discretized list of the contour, we then apply the Euclidean distance formula through KDTrees to find the closest point from the contour for a given point. Given the use of Euclidean distance to find the closest point on the decision boundary, the ```optimal_point()``` as described below only finds a single counterfactual explanation for the original instance without changing as few features as possible. 

This repository contains all of the notebooks used demonstrating the method as well as packaging the relevant files under the ```docs/files``` directory. 

As for our Juypter Notebooks, we have one documentation notebook ```documentation_notebook.ipynb``` that documents the functions and their relevant parameters, return types, and values. Under our ```docs/tests``` folder, we have several notebooks that not only tests the method, but we have a notebook where we compare the computational costs of the grid-based approach with our binary search method as described in the paper. Our original files folder consists of all the files provided by my supervisor, Professor Milad, throughout this research project. 

Link to the Paper: https://drive.google.com/file/d/1_V42KYoFrXqetrI7xDaXaS_0HrWQG3CD/view?usp=sharing

# Required Python Versions # 

Please check the ```requirements.txt``` for the python versions required for this repository. 


# Acknowledgement # 

Professor Milad Toutouchian supported my work during the duration of this project across 10 months from November 2024 to August 2025. His support has helped the initiation and progression of this project to this end. 





