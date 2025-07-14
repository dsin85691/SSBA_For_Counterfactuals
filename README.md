# Optimal Point for Counterfactual Explanations

As defined by Dandl and Molnar in their book "Interpretable Machine Learning," a counterfactual explanation can be defined as "the smallest change to the feature values that changes the prediction to a predefined output" (Dandl & Molnar 2019). 

Based on their definition, we define the following criteria for a counterfactual explanation. The first criterion is that the "prediction of the counterfactual must match the predefined outcome exactly." This means that we want to have the "minimal changes in the features" so that the predicted probability switches to the desired class. 

Another criterion is that "a counterfactual should be as similar as possible to the instance regarding feature values" (Dandl and Molnar 2019). The counterfactual should be close to the original instance and should also "change as few features as possible" (Dandl and Molnar 2019). 

Given the use of Euclidean distance to find the closest point on the decision boundary, the optimal_point() as described below only finds a single counterfactual explanation for the original instance without changing as few features as possible. 

This repository contains all of the required files for computing the counterfactual explanation  given a set of parameters: an labelled dataset 


# Meeting Notebook (May 30th, 2025) # 

The file ```meeting_notebook.ipynb``` contains all of the code needed for the meeting at Friday, May 30th, 2025. All of the required cells are part of this meeting notebook. Cells include a description of the problem statement, goals, and descriptions of used functions for the ```optimal_point()``` function. 
