# binaryhelpers

This module contains two files that contain many helper functions for creating, comparing and analyzing binary classification models.

The most common helpers used are:

### model_building_helpers.analyze_insp_opt:  
    - This function takes some data and analyzes the performance of a classificaiton model of your choosing.  It uses cross_validation to analyze the model's performance in multiple ways. It can handle all SKLearn classifiers as well as Pipelines 
    - as well 
    - Params: data: the dataframe of the data that contains your X and ys
    -       y_var: y variable you are using to train your model (0,1 for classification)
    -       X_vars: x variables you are using  to train your model
    -       classifier: the classifier you are using (sklearn classifer)
    -       add_dummy_vars: whether are not to add dummy variables to your x values
    -       return_solutions_pred: Whether or not to return the solutions of the cross validation you are applying
    -       folds: number of folds to specify for cross validation.  
    -       insp_cost: the cost of an inspection (for inspection_optimization)
    -       loss_avoidance: the benefit of finding a target variable
    -       return_estimators: whether or not to return the models in your dictionary
    -       mapper: a mapper variable for your x values, if you want to map them to a different set of fields
    -       verbose: whether or not to print out information as the model is being trained
    -       check_full_attr: whether or not to check_full_attr for feature importances.   This will add up importances for dummy variables based on the same field
    -       return_splitter: whether or not to return the splitter used for cross validation
    -       perm_imps: whether or not to return the permutation_importances of the models( this will make it take much longer)
    -       fit_params: optional fit_params you can pass into .fit method of your classifier

    - Returns:
    -       dictionary with multiple dataframes
    -       ar_curve_df: summary of the precision, recall and cost of inspections at each quantile from 1 to 100
    -       summary_df: summary of the cross validation with training and test scores
    -       feature_imps: the average of the feature importances using the classifier
    -       solutions_pred: the solutions and predicted solutions of the model
    -       estimators: a list of the estimators used in the cross_validation
    -       X_vars: since there is some manipulation of the order of the X variables we also return X_vars
    -       perm_imps: if specified will return permuration importances

### plot_funcs.compare_models:

    - Compares multiple models and returns a dictionary as well as the 3 curves comparing them.
    
    -   Params: 
    -        xlsx: the excel file you wish to save your data to, supply false if you don't want to save data output
    -        model_dict: optional supply of a existing model dictionary that contains specified necessary information about the model (minimum is model_dict[model][x][solutions_pred])
    -        colors: colors you want to use for your plots, if not supplied will use random colors
    -        return_figs: whether or not to return the figs for your plots
	-    **model_kwargs: kwarges to pass into analyze multiple models the wrapper around analyze_insp_opt that does all the work to train the model and generate anlaysis data

    -    returns:
    -        model_dict: the model analysis dictionary that it created or the one you supplied
    -        figs: 3 figures that are used to compare models ROC curve, PR curve and Lift chart
    

### plot_funcs.look_at_models:

    - Params: Data: data dataframe you are passing to this
    -          target: the target data you want to measure your comparison with
    -         score_list: columns from the model that indicator how your model scored

    Returns: agg_df: aggregate dataframe that returns the number of target variables in the top 20, bottom 20, top third, middle third and top third of scores
    -          figs: 3 figures Roc Curve, Lift Chart and Precision recall curve, comparing the performance of the models


Other functions include:

### model_building_helpers.logit_anlysis:

- A function used to analyze the results of a logit model from statsmodels, it will automatically remove highly correlated features using
variance inflation factors

### model_building_helpers.address_vif:

-A function that takes set of features and will eliminate highly correlated features using variance inflation factors
and a vif_threshold of your choosing.

### model_building_helpers.add_dummies:

- adds dummie variables given the data and a list of variables of your choosing



