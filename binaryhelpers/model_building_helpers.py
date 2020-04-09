from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import BaseCrossValidator
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.formula.api as sm
import os
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


def plot_corr(plot_data, title = 'title'):

    #function to plot a correlation coefficient.
    def corr_func(x, y, **kwargs):
        r = np.corrcoef(x, y)[0][1]
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(.2, .8), xycoords=ax.transAxes,
                    size = 20)

    # Create the pairgrid object
    grid = sns.PairGrid(data = plot_data, size = 3)

    # Upper is a scatter plot
    grid.map_upper(plt.scatter, color = 'red', alpha = 0.6)

    # Diagonal is a histogram
    grid.map_diag(plt.hist, color = 'red', edgecolor = 'black')

    # Bottom is correlation and density plot
    grid.map_lower(corr_func);
    grid.map_lower(sns.kdeplot, cmap = plt.cm.Reds)

    # Title for entire plot
    plt.suptitle(title, size = 36, y = 1.02)
    plt.show()

def remove_low_features(df, threshold = .001):
    reason_means = model_df[[col for col in model_df.columns if col.startswith('REASON')]].mean()
    reason_means = reason_means[(reason_means < threshold) | (reason_means == 1)].index.tolist()
    reason_means+=(reason_map_df[reason_map_df['reason_weight']==1].index.tolist())
    new_model_df = df.copy().drop(columns = reason_means)
    return new_model_df

def remove_categorical(df, unique_range = [7,15]):
    object_type = [col for col in test_model.columns if test_model[col].dtype == object]
    object_drop_ = test_model[object_type].nunique()
    object_drop_list = object_drop_[(object_drop_<unique_range[0]) | (object_drop_>unique_range[1])].index.tolist()
    new_model_df = df.copy().drop(columns = object_drop_list)
    return new_model_df

# def normalize_features(df, features = []):

def get_correlated_columns(df, threshold = 0.95, axis = 'columns'):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
    if axis == 'columns':
        to_drop = [column for column in upper.columns if any(upper[column]>threshold)]
    elif axis == 'index':
        to_drop = [ind for ind in upper.index if any(upper[index]>threshold)]

    return to_drop


def plot_feature_imps(clf, X, check_full_attr = True, only_data = False, split_str='_'):

#     Params: clf = classifer model
#             X = the x values of the model so we can plot their importance
#     Returns:
#             shows a plot of the importances
    if isinstance(clf,Pipeline):
        clf = clf.steps[-1][1]
    try:
        feature_imp = pd.Series(clf.feature_importances_, index = X.columns).sort_values(ascending = False)
    except AttributeError:
        feature_imp = pd.Series(clf.coef_[0], index = X.columns).sort_values(ascending = False)

    if check_full_attr:
        split_str_list1 = [index for index in feature_imp.index.values if index.count(split_str) >0]
        split_str_list2 = []
        for index in split_str_list1:
            col = index[:index.rindex(split_str)]
            split_str_list2.append(col)
            

        for index in split_str_list1:
            x = 0
            try:
                
                col = index[:index.rindex(split_str)]
                if split_str_list2.count(col)>1:
                    if col not in feature_imp.index.values:
                        feature_imp.loc[col] = feature_imp.loc[index]
                    else:
                        feature_imp.loc[col] = feature_imp.loc[col]+feature_imp.loc[index]
                    feature_imp.drop(index, axis = 0, inplace = True)
            except ValueError:
                continue
                
    feature_imp = feature_imp.sort_values(ascending = False)
    if only_data:
        return feature_imp
        
    fig16, ax16 = (plt.subplots(figsize = (15,40)))
    sns.barplot(x = feature_imp, y = feature_imp.index, ax = ax16)
    return fig16, ax16, feature_imp



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax

def add_dummies(df, dummies_list = ['CONSTRUCTIONTYPE', 'ROOFCOVERTYPE', 'DWELLINGTYPE']):
    for col in dummies_list:
        na = False
        if df[col].isnull().sum() > 0:
            na == True
        df = pd.concat([df, pd.get_dummies(df[col],dummy_na = na,prefix = col,drop_first = True)], axis = 1)
        df.drop(col, axis = 1, inplace = True)
    return df

def create_ar_curve(data,results_rows=101, output_var="model_output",target_var="Inspectiontarget"):
    increment = 1/(results_rows-1)
    results = pd.DataFrame(columns = ['percentile_from','percentile_to','score_from','score_to','hazards','inspections','precision','lift'])
    results['percentile_from'] = np.arange(0,1,increment)
    results['percentile_to'] = 1
    results['score_from'] = data[output_var].quantile(results['percentile_from']).values
    results['score_to'] = 1
    def add_haz_insps(row):
        row['hazards'] = (data[target_var]*(row['score_from']<=data[output_var])*(data[output_var]<=row['score_to'])).sum()
        row['inspections']=((row['score_from']<=data[output_var])*(data[output_var]<=row['score_to'])).sum()
        return row
    results = results.apply(add_haz_insps,axis = 1)
    results['recall'] = results['hazards']/data[target_var].sum()
    results['precision'] = results['hazards']/results['inspections']
    results['lift'] = results['precision']/results['precision'][0]-1
    
    # results = results.applymap(lambda x: np.round(x,3))
    return results
def model_scores(y_val,X_val,classifier,folds=5,insp_cost=30, loss_avoidance=350, return_solutions = False,fit_params={}):
    predictions = np.array(get_scores(X_val,y_val,classifier,folds))
    solutions_pred = pd.DataFrame(np.stack([np.array(y_val),predictions[:,1]],axis=1),columns = ['y_actual','y_perc'])
    final_df = create_ar_curve(solutions_pred, output_var = 'y_perc', target_var = 'y_actual')
    final_df['inspection_cost'] = np.multiply(final_df['inspections'],insp_cost)
    final_df['loss_avoidance'] = np.multiply(final_df['hazards'],loss_avoidance)
    final_df['value'] = final_df['loss_avoidance']-final_df['inspection_cost']
    if not return_solutions:
        return final_df
    else:
        return (final_df, solutions_pred)
    
def model_scores_reg(y_val,X_val,classifier,folds=5, return_solutions = False,fit_params={}):
    predictions = cross_val_predict(classifier,X_val,y_val,cv=folds,fit_params=fit_params)
    solutions_pred = pd.DataFrame(np.stack([np.array(y_val),predictions],axis=1),columns = ['y_actual','y_pred'])
    return solutions_pred

def model_importances(y_val,X_val,classifier):
    score_dict = cross_validate(clf,X_val,y_val, cv=5,return_estimator=True)
    imp_list = []
    for model in score_dict['estimator']:
        imp = plot_feature_imps(model,X_val,only_data=True)
        imp_list.append(imp)
    return imp_list

def analyze_insp_opt(data, y_var,X_vars,classifier,add_dummy_vars=True, return_solutions_pred = True,folds=5, insp_cost = 30, loss_avoidance=350, return_estimators = False,mapper={},verbose = False,check_full_attr=False,return_splitter=True,perm_imps=False,fit_params={}):


    # This function takes some data and analyzes the performance of a classificaiton model of your choosing.  It uses cross_validation to analyze the model's performance in multiple ways.  It can handle all SKLearn classifiers as well as Pipelines 
    # as well 
    # Params: data: the dataframe of the data that contains your X and ys
    #       y_var: y variable you are using to train your model (0,1 for classification)
    #       X_vars: x variables you are using  to train your model
    #       classifier: the classifier you are using (sklearn classifer)
    #       add_dummy_vars: whether are not to add dummy variables to your x values
    #       return_solutions_pred: Whether or not to return the solutions of the cross validation you are applying
    #       folds: number of folds to specify for cross validation.  
    #       insp_cost: the cost of an inspection (for inspection_optimization)
    #       loss_avoidance: the benefit of finding a target variable
    #       return_estimators: whether or not to return the models in your dictionary
    #       mapper: a mapper variable for your x values, if you want to map them to a different set of fields
    #       verbose: whether or not to print out information as the model is being trained
    #       check_full_attr: whether or not to check_full_attr for feature importances.   This will add up importances for dummy variables based on the same field
    #       return_splitter: whether or not to return the splitter used for cross validation
    #       perm_imps: whether or not to return the permutation_importances of the models( this will make it take much longer)
    #       fit_params: optional fit_params you can pass into .fit method of your classifier

    # Returns:
    #       dictionary with multiple dataframes
    #       ar_curve_df: summary of the precision, recall and cost of inspections at each quantile from 1 to 100
    #       summary_df: summary of the cross validation with training and test scores
    #       feature_imps: the average of the feature importances using the classifier
    #       solutions_pred: the solutions and predicted solutions of the model
    #       estimators: a list of the estimators used in the cross_validation
    #       X_vars: since there is some manipulation of the order of the X variables we also return X_vars
    #       perm_imps: if specified will return permuration importances

    return_dict = {}
    if add_dummy_vars:
        obj_vars = [col for col in X_vars if data[col].dtype == object]
        if len(obj_vars)>1:
            if verbose:
                print("Adding Dummies")
            data = add_dummies(data, dummies_list = obj_vars)
            if verbose:
                print("Dummies Added")
    new_X_vars = [var for var in data.columns if var in X_vars or '_'.join(var.split('_')[:-1]) in X_vars]
    X = data[new_X_vars]
    y = data[y_var]
    return_dict['X_vars'] = [mapper[var] if var in mapper.keys() else var for var in new_X_vars]
    if verbose:
        print("Shape of y is {} rows".format(str(y.shape[0])))
        print("Shape of X is {} rows and {} columns".format(str(X.shape[0]), str(X.shape[1])))
    if verbose:
        print("Training Models with Cross Validation")
    if type(folds)==int:
        folds = StratifiedKFold(n_splits=folds)
    if not isinstance(folds,BaseCrossValidator):
        raise TypeError("Folds must be of instance BaseCrossValidator")

    cv_results = cross_validate(classifier,X,y,cv=folds,return_estimator=True, return_train_score=True, scoring=['roc_auc','average_precision','accuracy'],fit_params=fit_params)
    estimators = cv_results.pop('estimator',None)
    if verbose:
        print("Cross Validation completed")
    if verbose:
        print("Getting model scores")
    ar_curve_df = model_scores(y,X,estimators,folds=folds,return_solutions=return_solutions_pred,fit_params=fit_params)
    if verbose:
        print("Model scores completed")
    if return_solutions_pred:
        return_dict['ar_curve_df'] = ar_curve_df[0]
        return_dict['solutions_pred'] = ar_curve_df[1]
    else:
        return_dict['ar_curve_df'] = ar_curve_df
    
    if return_estimators:
        return_dict['estimators'] = estimators
    return_dict['summary_df'] = pd.DataFrame(cv_results)
    try:
        if verbose:
            print("Getting Feature Imps.")
        feature_imps_list = [plot_feature_imps(model,X,check_full_attr=check_full_attr,only_data=True) for model in estimators]
        new_df = pd.concat(feature_imps_list,axis = 1,sort=True)    
        new_df.index = pd.Series(new_df.index).apply(lambda x: x if x not in mapper.keys() else mapper[x])
        feature_imps = new_df.mean(axis=1).sort_values(ascending=False)
        return_dict['feature_imps'] = feature_imps
        if verbose:
            print("Feature_imps acquired")
    except Exception as e:
        print(e)
        print("No Feature Importances available in for this classifier")
    if perm_imps:
        if verbose:
            print("Gettting Perm Imps")
        perm_importances = get_perm_imp(X,y,estimators,folds)
        return_dict['perm_imps'] = perm_importances
        
        if verbose:
            print("Perm Imps Acquired")
    return return_dict

def analyze_regression(data, y_var,X_vars,regressor,add_dummies=True, return_estimators=True,return_solutions_pred = False,folds=5,verbose=True,fit_params={},check_full_attr=True):
    return_dict = {}
    if add_dummies:
        obj_vars = [col for col in X_vars if data[col].dtype == object]
        if len(obj_vars)>0:
            if verbose:
                print("Adding Dummies")
            data = add_dummies(data, dummies_list = obj_vars)
            if verbose:
                print("Dummies Added")
            new_X_vars = [var for var in data.columns if var in X_vars or '_'.join(var.split('_')[:-1]) in X_vars]
        else:
            new_X_vars = X_vars
    X = data[new_X_vars]
    y = data[y_var]
    if verbose:
        print("Shape of y is {} rows".format(str(y.shape[0])))
        print("Shape of X is {} rows and {} columns".format(str(X.shape[0]), str(X.shape[1])))
    if return_solutions_pred:        
        if verbose:
            print("Getting model scores")
        solutions_pred = model_scores_reg(y,X,regressor,folds=folds,return_solutions=return_solutions_pred,fit_params=fit_params)
        if verbose:
            print("Model scores completed")
        return_dict['solutions_pred'] = solutions_pred
    if verbose:
        print("Training Models with Cross Validation")
    cv_results = cross_validate(regressor,X,y,cv=folds,return_estimator=True, return_train_score=True, scoring=['neg_mean_squared_error'],fit_params=fit_params)
    if verbose:
        print("Cross Validation completed")
    estimators = cv_results.pop('estimator',None)
    if return_estimators:
        return_dict['estimators'] = estimators
    return_dict['summary_df'] = pd.DataFrame(cv_results)
    try:
        if verbose:
            print("Getting Feature Imps.")
        feature_imps_list = [plot_feature_imps(model,X,check_full_attr=check_full_attr,only_data=True) for model in estimators]
        new_df = pd.concat(feature_imps_list,axis = 1,sort=True)    
        feature_imps = new_df.mean(axis=1).sort_values(ascending=False)
        return_dict['feature_imps'] = feature_imps
        if verbose:
            print("Feature_imps acquired")
    except Exception as e:
        print(e)
        print("No Feature Importances available in for this classifier")
    return return_dict
                                            
def logit_analysis(data, target_var, predictor_vars, vif_threshold = 5, mapper = {}, pvalue=False):

    # This function uses the logit function from statsmodels and outputs a DataFrame with metrics for how each predictor var 
    # relates to the target var

    # Params: data: dataframe you wish to explort
    #       target_var: the target variable that you want your logit to predict (must be a column in the data)
    #       predictor_vars: the columns of all the predictor vars 
    #       vif_threshold: the threshold of vif that you want to have to make sure you remove from the logit logit_analysis
    #       mapper: a mapper for your columns in case you have pretty names otherwise there will be columns with the same names

    # Returns dictionary of
    #       metrics: dataframe all assuring that the VIF < vif_threshold
    #       high_vif_dict: dictionary of variable 
    #       logit: the regression object so you can look at the result
    
    data = data[[target_var]+predictor_vars].dropna().astype(np.int64)
    data.columns = [col.replace(' ','_') for col in data.columns]
    X_values = data[[var.replace(' ','_') for var in predictor_vars]]
    Y_values = data[target_var.replace(' ','_')]
    X_vals = add_constant(X_values)


    vif = [variance_inflation_factor(X_vals.values,i) for i in range(X_vals.shape[1])]
    i=0
    high_vif_dicts = []
    while max(vif[1:]) > vif_threshold:
        append_dict={}
        col_to_remove = X_vals.columns[1:][vif[1:].index(max(vif[1:]))]
        append_dict['orig_feature'] = col_to_remove
        if col_to_remove in mapper.keys():
            append_dict['mapped_feature'] = mapper[col_to_remove]
        append_dict['vif'] = (max(vif[1:]))
        high_vif_dicts.append(append_dict)
        X_vals = X_vals.drop(columns=col_to_remove)
        vif = [variance_inflation_factor(X_vals.values,i) for i in range(X_vals.shape[1])]
        i+=1
        #Lets make sure we don't get stuck here
        if i%5==0:
            print("{} features removed due to high VIF".format(str(i)))
        if i >30:
            break
            

    X_string = '+'.join(X_vals.columns[1:])
    reg = sm.logit("""{} ~ {}""".format(target_var,X_string),data=data).fit()
    indexes = [i for i in range(0,len(reg.pvalues))]
    if pvalue:
        cur_index = list(reg.pvalues.index)
        remove_cols = list(reg.pvalues[1:][reg.pvalues[1:]>pvalue].index)
        indexes = [index for index in range(0,len(cur_index)) if cur_index[index] not in remove_cols]
        for col in remove_cols:
            append_dict = {}
            append_dict['orig_feature']=col
            append_dict['pvalue']=reg.pvalues[col]
            if col in mapper.keys():
                append_dict['mapped_feature'] = mapper[col]
            high_vif_dicts.append(append_dict)
        X_vals = X_vals.drop(columns=remove_cols)
        X_string = '+'.join(X_vals.columns[1:])
        reg = sm.logit("""{} ~ {}""".format(target_var,X_string),data=data).fit()
    metrics = pd.DataFrame()
    metrics['orig_feature'] = X_vals.columns
    metrics['mapped_feature'] = [mapper[col] if col in mapper.keys() else col for col in X_vals.columns]
    metrics['vif'] = [vif[i] for i in indexes]
    metrics['pvalues'] = round(reg.pvalues,3).values
    metrics['odds'] = np.exp(reg.params.values)
    metrics['count'] = X_values.shape[0]
    metrics.loc[1:,'count'] = X_vals[X_vals.columns[1:]].sum().values
    metrics['target_count'] = Y_values.sum()
    metrics.loc[1:,'target_count'] = data[data[target_var]==1][X_vals.columns[1:]].sum().values
    metrics['perc'] = metrics['target_count']/metrics['count']
    metrics['odds_conf_int_.025'],metrics['odds_conf_int_.975'] = np.exp(reg.conf_int()[0].values), np.exp(reg.conf_int()[1].values)
    return_dict = {}
    return_dict['logit'] = reg
    return_dict['metrics'] = metrics
    return_dict['high_vif_features'] = high_vif_dicts
    return return_dict

def summarize_dist(data,field,target_field,thresh=0):
    
    cols = ['{}_cnt'.format(str(field)),'{}_dist'.format(str(field)),'{}_cnt'.format(str(target_field)),'{}_dist'.format(str(target_field))]
    field_cnt = data[field].value_counts()
    field_dist = data[field].value_counts()/data.shape[0]
    target_cnt = data[data[target_field].astype(int)==1][field].value_counts()
    target_dist = target_cnt/field_cnt
    summarize_df = pd.concat([field_cnt,field_dist,target_cnt,target_dist],axis=1, sort=False)
    summarize_df.columns=cols
    return summarize_df[summarize_df[cols[1]]>thresh]


def make_long(df, index, columns, values):
    not_cols = [columns,values]
    long_df = df.pivot(index=index, columns=columns, values=values).reset_index()
    return_df = df[[col for col in df.columns if col not in not_cols]].drop_duplicates().merge(long_df, how='inner',
                                                                                              left_on=index,right_on=index)
    return return_df

def analyze_multiple_models(data, preds, X_cols, models, multi_x = False, **model_kwargs):
    ## wrapper around analyze_insp_opt, that allows you to run it on multiple models and returns a dictionary of how those models performed.

    i=0
    x=0
    all_model_dict = {}
    for model in models:
        model_analyze_dicts = {}
        if not multi_x:
            for pred in preds:
                model_dict = analyze_insp_opt(data,pred, X_cols, model, **model_kwargs)
                model_analyze_dicts[pred] = model_dict
        else:
            for pred in preds:
                for cols in X_cols:
                    model_dict = analyze_insp_opt(data,pred,cols,model,**model_kwargs)
                    x+=1
                    model_analyze_dicts["{}_x{}".format(pred,str(x))]=model_dict
        i+=1
        if isinstance(model,Pipeline):
            model = model.steps[-1][1]
        all_model_dict['Model_{}_{}'.format(str(i),str(type(model).__name__))]=model_analyze_dicts

    return all_model_dict

def add_bin_col(data, col, bins=5):
    if isinstance(bins,list):
        bin_data, retbins = pd.cut(data[col], bins, retbins=True)
        labels = ['{} ({} - {})'.format(col,str(retbins[i]),str(retbins[i+1])) for i in range(len(retbins)-1)]
        new_bin_data = pd.cut(data[col], bins, labels=labels)
    else:
        bin_data, retbins = pd.qcut(data[col], bins, retbins=True)
        labels = ['{} ({} - {})'.format(col,str(retbins[i]),str(retbins[i+1])) for i in range(len(retbins)-1)]
        new_bin_data = pd.qcut(data[col], bins, labels=labels)

    return new_bin_data


def create_lift_gains(data,pred,target,split=100,labels=False,**cut_kwargs):
    if isinstance(split,list):
        if not labels:
            data['split'] = pd.cut(data[pred],bins=split,right=False,include_lowest=True,duplicates='drop',**cut_kwargs)
        else:
            data['split'] = pd.cut(data[pred],bins=split,labels=labels,right=False,include_lowest=True,duplicates='drop',**cut_kwargs)
    else:
        data['split'] = pd.qcut(data[pred],split)
    new_df = data.groupby('split').agg({target:['sum','count']})
    new_df.columns = ['_'.join(x) for x in new_df.columns.ravel()]
    new_df.columns = ['N Targets', 'N Obs']
    new_df['Perc Targets'] = new_df['N Obs']/new_df['N Obs'].sum()
    new_df.sort_index(ascending=False, inplace=True)
    new_df['percentiles'] = new_df['Perc Targets'].cumsum()
    new_df['Cumulative Obs'] = new_df['N Obs'].cumsum()
    new_df['Cumulative Targets'] = new_df['N Targets'].cumsum()
    new_df['Target Dist'] = new_df['N Targets']/new_df['N Obs']
    new_df['Non Target Dist'] = 1 - new_df['Target Dist']
    new_df['Non Target Gains'] = (new_df['N Obs']-new_df['N Targets']).cumsum()/(new_df['N Obs']-new_df['N Targets']).cumsum().max()
    new_df['Gains'] = new_df['Cumulative Targets']/new_df['Cumulative Targets'].max()
    new_df['Lift'] = new_df['Gains']/new_df['percentiles']
    new_df['KS'] = new_df['Gains'] - new_df['Non Target Gains']
    return new_df

def address_vif(X_values,vif_threshold=5):
    X_vals = add_constant(X_values)
    vif = [variance_inflation_factor(X_vals.values,i) for i in range(X_vals.shape[1])]
    i=0
    high_vif_dicts = []
    while max(vif[1:]) > vif_threshold:
        append_dict={}
        col_to_remove = X_vals.columns[1:][vif[1:].index(max(vif[1:]))]
        append_dict['orig_feature'] = col_to_remove
        append_dict['vif'] = (max(vif[1:]))
        high_vif_dicts.append(append_dict)
        X_vals = X_vals.drop(columns=col_to_remove)
        vif = [variance_inflation_factor(X_vals.values,i) for i in range(X_vals.shape[1])]
        i+=1
        #Lets make sure we don't get stuck here
        if i%5==0:
            print("{} features removed due to high VIF".format(str(i)))
        if i >30:
            break
    map_dict = dict(zip(X_vals.columns,vif))
    return high_vif_dicts,map_dict,list(X_vals.columns[1:])

def get_scores(X,y,ordered_models,splitter):
    """Params:
            X: X variable for the ordered models
            y: y variable (target) variable for the ordered models
            ordered_models: the ordered list of models fed into cross validation
            splitter: the sklearn implementation of a cross validation splitter
        Returns:
            finals: dataframe of the scores of the model accross all splits
    """
    i=0
    scores = []
    for train_index,test_index in splitter.split(X,y):
        model = ordered_models[i]
        score = pd.DataFrame(model.predict_proba(X.loc[test_index]),columns = np.unique(y))
      
        score.index = test_index
        scores.append(score)
        i+=1
    finals = pd.concat(scores)
    finals['pred'] = [np.unique(y)[x] for x in finals.values.argmax(axis=1)]
    return finals.sort_index()

def get_perm_imp(X,y,ordered_models,splitter,scoring='roc_auc'):
    """Params:
            X: X variable for the ordered models
            y: y variable (target) variable for the ordered models
            ordered_models: the ordered list of models fed into cross validation
            splitter: the sklearn implementation of a cross validation splitter
            scoring: what scoring method you want to apply to your permutation importances
        Returns:
            perm_imps: dataframe of the permutation importances for all variables
    """
    i=0
    perm_imp_list = []
    for train_index,test_index in splitter.split(X,y):
        model = ordered_models[i]
        bunch = permutation_importance(model, X.loc[test_index],y.loc[test_index],scoring=scoring,n_jobs=12,random_state=42)
        perm_imp_list.append([pd.Series(bunch['importances_mean']),pd.Series(bunch['importances_std'])])
    perm_imps = pd.DataFrame()
    perm_imps['features'] = X.columns
    test = [perm_imp[0] for perm_imp in perm_imp_list]
    perm_imps['perm_imp_means']= pd.concat([perm_imp[0] for perm_imp in perm_imp_list],axis=1).mean(axis=1)
    perm_imps['perm_imps_std'] = pd.concat([perm_imp[1] for perm_imp in perm_imp_list],axis=1).std(axis=1)
    return perm_imps.sort_values(by='perm_imp_means',ascending=False)

def write_analysis_data(writer,model_dict,sheet_prefix,write_solutions_pred = False):

    model_dict['summary_df'].to_excel(writer, sheet_name='{}_summary'.format(sheet_prefix))
    model_dict['ar_curve_df'].to_excel(writer, sheet_name='{}_arcurve'.format(sheet_prefix))
    if 'feature_imps' in model_dict.keys():
        model_dict['feature_imps'].to_excel(writer, sheet_name='{}_feature_imps'.format(sheet_prefix))
    if 'perm_imps' in model_dict.keys():
        model_dict['perm_imps'].to_excel(writer, sheet_name='{}_perm_imps'.format(sheet_prefix))
    if write_solutions_pred and 'solutions_pred' in model_dict.keys():
        model_dict['solutions_pred'].to_excel(writer,sheet_name='{}_solutions'.format(sheet_prefix))