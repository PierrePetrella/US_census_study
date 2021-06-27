### Imports ###

# Visualisation 
import matplotlib.pyplot as plt
import seaborn as sns

# Data Manipulation
import random
import pandas as pd
import numpy as np
import json
import ast

# Statistical Tests
from scipy.stats import ttest_ind

# Modelling
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

import xgboost as xgb

### Functions ###

# Univariate Study of Continuous features
def univariate_continuous_viz(df,continuous_features ):
    """
    Univariate Study of Continuous features
    Params:
        df : train dataframe
        continuous_features : list
    """
    
    print ("The continuous features are: {}".format(continuous_features))
    # For each feature show a histogram to visualize 
    n_bins=20

    # generate plots of histograms
    fig, axs = plt.subplots(int(len(continuous_features)/4), 4, sharey=False, sharex=False, tight_layout=True, figsize=(20,8))
    fig.suptitle('Distribution of continuous features')
    axs = axs.ravel()
    for idx,ax in enumerate(axs):
        ax.grid()
        ax.set_title("Distribution of {}".format(continuous_features[idx]))
        #ax.hist(df[numerical_features[idx]], bins=n_bins, color=["red"])
        sns.distplot(df[continuous_features[idx]],ax=ax, bins=n_bins)
    
    # Compute basic statistics: count, mean, std, min, max and quartiles
    df_stats = df[continuous_features].describe()
    # Compute skewness for more information on the distribution
    df_stats.loc["skew"] = df.skew()
    # Compute kurtosis, another indicator of distribution
    df_stats.loc["kurtosis"] = df.kurt()
    
    return df_stats

# Bivariate Study of Continuous features
def bivariate_continuous_viz(df, continuous_features, target, seed):
    """
    Bivariate Study of Continuous features
    Params:
        df : train dataframe
        continuous_features : list
        target : string (name of the target column) 
        seed : int
    """
    
    df_under_50 = df[df[target]==' - 50000.']
    df_over_50 = df[df[target]==' 50000+.']

    fig, axs = plt.subplots(int(len(continuous_features)/4), 4, sharey=False, sharex=False, tight_layout=True, figsize=(20,8))
    fig.suptitle('Bivariate study of continuous features')
    axs = axs.ravel()
    for idx,ax in enumerate(axs):
        ttest = ttest_ind(random.sample(df_over_50[continuous_features[idx]].values.tolist(), 300),
                                        random.sample(df_under_50[continuous_features[idx]].values.tolist(), 300),
                                        equal_var=False)
        ax.grid()
        ax.set_title('{f} vs. {t} \n mean for <50 = {l} \n mean for >50 = {m} \n t-test: statistic={s}, pvalue={p}'
                                                                             .format(f = continuous_features[idx],
                                                                              t = target,
                                                                              l = df_under_50[continuous_features[idx]].mean(),
                                                                              m = df_over_50[continuous_features[idx]].mean(),
                                                                              s=round(ttest.statistic, 2),
                                                                              p=round(ttest.pvalue, 3)))
       
        ax.set_xlabel(target)
        ax.boxplot ([df_under_50[continuous_features[idx]], df_over_50[continuous_features[idx]]], labels=[' - 50000.',' 50000+.'])
    return


# Compute the correlation matrix
def correlation_continuous_viz(df, continuous_features):
    """
    Compute the correlation matrix and print a graph
    Params:
        df : train dataframe
        continuous_features : list
    """
    
    corr = df[continuous_features].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap,annot=True, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    

    # Display categorical univariate content of each one feature
def categorical_univariate_plot(dataframe, categorical_feature):
    """
    Generate a Countplot for a categorical feature 
    Params:
        df : train dataframe
        categorical_feature : string
    """
    
    plt.figure(figsize=(20,max(5,len(dataframe[categorical_feature].unique())/3)))
    ax = sns.countplot(y=categorical_feature, data=dataframe)
    ax.set_title("Count plot of '{}'".format(categorical_feature))
    return plt.show()

 # Display categorical univariate content of each one feature
def univariate_categorical_viz(df, categorical_features):
    """
    Generate a set Countplots for a categorical features and print statistics
    Params:
        df : train dataframe
        categorical_features : list
    """
    for feature in categorical_features:
        print("Distribution of the:'{}' categories".format(feature))
        print("There are {} distinct categories".format(len(df[feature].unique())))  
        print("Most common label is '{m}' with {p}%"
              .format(m= df[feature].mode()[0],
                      p=round(100*len(df.loc[df[feature] == df[feature].mode()[0]])/len(df))))
        categorical_univariate_plot(df,feature) 
        
        
def categorical_bivariate_plot(df, categorical_feature, target):
    """
    Generate a Countplot for a categorical feature 
    Params:
        df : train dataframe
        categorical_feature : string
        target : string
    """
    
    plt.figure(figsize=(20,max(5,len(df[categorical_feature].unique())/3)))
    ax = sns.countplot(y=categorical_feature,hue=target, data=df)
    ax.set_title("Count plot of '{}'".format(categorical_feature))
    return plt.show()

 # Display categorical bivariate content of each one feature
def bivariate_categorical_viz(df, categorical_features, target):
    """
    Generate Countplots for a categorical features  
    Params:
        df : train dataframe
        categorical_feature : list
        target : string
    """
    for feature in categorical_features:
        print("Comparing '{}' feature with the target '{}'".format(feature, target)) 
        categorical_bivariate_plot(df,feature, target)  
        
        
# Format data properly before feature engineering
def dataCleaning(df,target):
    """
    Format data properly before feature engineering
    Params:
        df : dataframe (train or test)
        target : string
    """
    print ("Number of lines in the dataset before dropping duplicates:",len(df))

    # 1) Drop duplicates
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    print ("Number of lines in the dataset after dropping duplicates: {}".format(len(df)))

    # 2)onvert target class to binary
    df[target] = df[target].map({' - 50000.': 0, ' 50000+.': 1})

    # 3) Imputing missing values

    # First, we replace question marks with nulls
    df = df.replace(' ?', np.nan)

    # Let us display which columns have null values:
    print("Columns which have missing values before imputation:")
    print(df.isnull().sum().sort_values(ascending=False).head(8))

    # impute nulls with the mode
    for cols in df.columns:
        df[cols] = df[cols].fillna(df[cols].mode()[0])

    return df

def featureEngineering(df_train, df_test, categorical_features, target):
    """
    Apply feature preparation to train_data
    Params:
        df_train : train dataframe
        df_test : test dataframe
        categorical_features : list
        target : string
    """
    df_train = dataCleaning(df_train.drop(columns="instance weight"), target)
    df_test = dataCleaning(df_test.drop(columns="instance weight"), target)

    # Creating training and test features and target variable

    X_train, y_train = df_train.drop(columns=[target]), \
    df_train[[target]].rename(columns={target : "y"})["y"] # train

    X_test, y_test = df_test.drop(columns=[target]), \
    df_test[[target]].rename(columns={target : "y"})["y"] # test

    # encoding categorical data
    le = LabelEncoder()
    for cols in categorical_features:
        X_train[cols] = le.fit_transform(X_train[cols])
        X_test[cols] = le.transform(X_test[cols])

    # standard scaling features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, y_train, X_test_scaled, y_test


def trainLogisticRegression(X_train, y_train):
    """
    Train LogisticRegression
    Params:
        X_train : train dataframe
        y_train : test dataframe
    """
    # Applying the Logistic Regression algorithm
    print("Logistic Regression Model: \n")
    logreg = LogisticRegression(solver="lbfgs") # define model
    score_logreg = cross_val_score(logreg, X_train, y_train, cv=5) # obtain cross validation score
    print("CV scores = {}".format(score_logreg))
    print("Mean CV score = {}".format(round(np.mean(score_logreg), 3)))
    model_logreg = logreg.fit(X_train, y_train) # fit to training
    logreg_importance = model_logreg.coef_[0]
    print("Feature importances:")
    for i,v in enumerate(logreg_importance):
        print(X_train.columns[i] + ' Score: %.5f' % (v))

    return model_logreg

def trainRandomForest(X_train, y_train, random_seed):
    """
    Train RandomForestClassifier
    Params:
        X_train : train dataframe
        y_train : test dataframe
        random_seed : int
    """


    print("Random Forest Classifier Model: \n")
    rf = RandomForestClassifier()
    
    random_grid = {'n_estimators':[int(x) for x in np.linspace(start = 5, stop = 30 , num = 5)],
                   'max_features': ['auto', 'sqrt'],
                   'max_depth': [int(x) for x in np.linspace(5, 30, num = 1)],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   'bootstrap': [True, False]}
    print("Random search parameters = {}".format(str(random_grid)))

    
    random_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, \
                                    cv = 5, verbose=1, random_state=random_seed)
    # Fit it to the data
    random_search.fit(X_train,y_train)
    print("Tuned Decision Tree Parameters: {}".format(random_search.best_params_))
    print("Best score is {}".format(random_search.best_score_))
    
    best_params = random_search.best_params_
    print("Best params are {}".format(best_params))
    model_rf = RandomForestClassifier(n_estimators= best_params["n_estimators"], \
                           min_samples_split= best_params["min_samples_split"], \
                           min_samples_leaf= best_params["min_samples_leaf"], 
                           max_features= best_params["max_features"], \
                           max_depth= best_params["max_depth"], \
                           bootstrap = best_params["bootstrap"]).fit(X_train, y_train)
    
    
    print("Feature importances:")
    for i, v in enumerate(random_search.best_estimator_.feature_importances_):
        print(X_train.columns[i] + ' Score: %.5f' % (v))

    return model_rf


def trainXGBoost(X_train,y_train, seed):
    """
    Train XGBoostClassifier
    Params:
        X_train : train dataframe
        y_train : test dataframe
        random_seed : int
    """
    
    
    # A parameter grid for XGBoost
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'n_estimators': range(10, 40, 1),
        'learning_rate': [0.1, 0.01, 0.05]
    }
    print("XGBoost Classifier Model: \n")
    print("Random search parameters = {}".format(str(params)))
    
    
    # fit model to training data
    model_xgb = xgb.XGBClassifier(objective = "binary:logistic", eval_metric = "logloss",)
    
    random_search = RandomizedSearchCV(model_xgb, params,
                                       n_iter=5, scoring='roc_auc', 
                                       cv=5, verbose=1,
                                       random_state=seed )
    random_search.fit(X_train, y_train)
    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    best_params =random_search.best_params_
    model_xgb = xgb.XGBClassifier(min_child_weight=best_params['min_child_weight'],
                                    gamma=best_params['gamma'],
                                    subsample=best_params['subsample'],
                                    colsample_bytree=best_params['colsample_bytree'],
                                    max_depth=best_params['max_depth'],
                                    n_estimators=best_params['n_estimators'],
                                    learning_rate=best_params['learning_rate']).fit(X_train, y_train)
    
    
    print("Feature importances:")
    for i, v in enumerate(random_search.best_estimator_.feature_importances_):
        print(X_train.columns[i] + ' Score: %.5f' % (v))

    return model_xgb



def model_assesment(model, X_test, y_test, model_name):
    
    """
    Evaluating a model on a testing set
    Params:
        model : model
        X_test : train dataframe
        y_test : test dataframe
        model_name : string
    """
    y_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories = ['<50k','>50k']
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    plt.figure()
    sns.heatmap(cf_matrix, annot=labels, fmt="",yticklabels=categories,xticklabels=categories, cmap='Blues')
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix for {} model".format(model_name))
    #plt.show()

    #Metrics for Binary Confusion Matrices
    
    # Accuracy is sum of diagonal divided by total observations
    # It represents the number of correct predictions over the total number of predictions
    accuracy  = np.trace(cf_matrix) / float(np.sum(cf_matrix))
   
    # Precision is the number of true positives over all the observations predicted positive
    precision = cf_matrix[1,1] / sum(cf_matrix[:,1])
    # Recall is the number of true positives over all the positives
    recall    = cf_matrix[1,1] / sum(cf_matrix[1,:])
    # f1_score is a combination of precision and recall
    f1_score  = 2*precision*recall / (precision + recall)
    
    #print( "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}"
    #  .format(accuracy,precision,recall,f1_score))
    
    return accuracy, precision, recall, f1_score


def compare_models(model_logreg, model_rf, model_xgb, X_test, y_test):
    """
    Evaluating all models on a testing set and compare them
    Params:
        model_logreg: model
        model_rf: model
        model_xgb: model
        X_test : train dataframe
        y_test : test dataframe
    """
    logistic_regression_scores = model_assesment(model_logreg, X_test, y_test, "logistic regression")
    random_forest_scores = model_assesment(model_rf, X_test, y_test, "random forest")
    xgboost_scores = model_assesment(model_xgb, X_test, y_test, "XGBoost")
    
    data = [logistic_regression_scores,random_forest_scores,xgboost_scores]
    index = ["Logistic Regression", "Random Forest Classifier","XGBoost"]
    columns= ['accuracy', 'precision', 'recall', 'f1_score']
    
    return pd.DataFrame(data, columns=columns, index = index)


