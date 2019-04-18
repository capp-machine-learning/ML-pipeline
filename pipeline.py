'''
Pipeline for Machine Learning Analysis.

Si Young Byun (syb234)
'''
import os
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score

# Read Data

def read_data(filename):
    '''
    Read a dataset and print a short summary of the data.
    Return a dataframe of the dataset.
    Input:
    - filename: (string) the directory of the dataset
    Output:
    - df: (pandas dataframe) dataframe of the dataset
    '''
    _, ext = os.path.splitext(filename)

    if ext == '.csv':
        df = pd.read_csv(filename, index_col=0)
    elif ext == '.xls':
        df = pd.read_excel(filename, header=1)

    try:
        print("############################################################\n")
        print("Data Shape: {}\n".format(df.shape))
        print("Descritive Statistics:\n\n{}\n".format(df.describe()))
        print("############################################################\n")

        return df

    except UnboundLocalError:
        print("Failed to read the data! \nPlease check the filename.")


# Explore Data

def view_variable_dist(df, variable):
    
    val_count = df[variable].value_counts()
    
    print("Count of values in {}:\n\n{}\n".format(variable, val_count))
    ax = val_count.plot('bar', rot=0, figsize=(15, 5))
    ax.set(xlabel=variable, ylabel="Count")
    plt.show()


def find_var_with_missing_values(df):
    '''
    Find variables with missing data and return those in a list.
    Input:
    - df: (pandas dataframe) dataframe of the dataset
    Output:
    - nan_vars: (list) list of the name of the variables with missing values
    '''
    nan = df.isna().sum()
    nan_perc = round(100 * nan / len(df.index), 2)
    nan_df = pd.concat([nan, nan_perc], axis=1)
    nan_df = nan_df.rename(columns = {0: 'NaN', 1: 'Percent of NaN'})
    nan_df = nan_df.sort_values(by=['Percent of NaN'], ascending=False)
    only_nan_df = nan_df[nan_df['NaN'] > 0]
    nan_vars = only_nan_df.index.tolist()

    print(nan_df)
    print("\nThe following variables have missing values: {}".format(nan_vars))

    message = "\n- {} has {} missing values, which are {}% of the entire data"

    for var in nan_vars:
        num = only_nan_df.loc[var][0]
        perc = only_nan_df.loc[var][1]
        print(message.format(var, num, perc))

    return nan_vars


def generate_boxplots(df, columns):
    '''
    Given the dataset and columns to examine, output one boxplot for
    each variable.
    Input:
    - df: (pandas dataframe) dataframe of the dataset
    - columns: (list) the name of the columns/variables
    Output:
    - boxplots
    '''
    for column in columns:
        
        fig, ax = plt.subplots(figsize=(15, 5))
        ax = sns.boxplot(x=df[column])

        # if kurtosis is beyond -3 and 3, log scale the x axis.
        if abs(df[column].kurt()) > 3:
            ax.set_xscale('log')

    plt.show()


def generate_corr_heatmap(df):

    # compute correlation
    corr = df.corr()
    
    # generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # create figure and plot
    f, ax = plt.subplots(figsize=(15, 5))
    
    # Generate a diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap, linewidths=.5)

    plt.show()


def find_iqr_outliers(df, column, weight=1.5):

    data = df[column]
    quantile_25, quantile_75 = np.percentile(data, [25, 75])
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest = quantile_25 - iqr_weight
    highest = quantile_75 + iqr_weight
    outlier_ind = np.where((data < lowest) | (data > highest))
        
    return outlier_ind


def visualize_outliers(df, columns):
    
    for column in columns:
        f, ax = plt.subplots(figsize=(15, 5))
        df.iloc[find_iqr_outliers(df, column)][column].hist(bins=25)
        plt.xlabel(column)
    
    plt.show()

# Pre-process Data

def impute_missing_data(df, columns):

    for column in columns:
        if abs(df[column].kurt()) > 3:
            cond = df[column].median()
            print("For {}, median is selected.".format(column))
        else:
            cond = df[column].mean()
            print("For {}, mean is selected.".format(column))
        estimate = round(cond)
        df[column] = df[column].fillna(estimate)

    print("Imputation completed!")

def drop_variable(df, columns):

    try:
        df.drop(labels=columns, axis=1, inplace=True)

    except:
        print("Unable to drop {}.".format(columns))

    else:
        print("{} are successfully dropped from the dataset".format(columns))


# Generate Features/Predictors

def discretize_variable(df, variable, bin_number, labels=None):

    bin_name = variable + "_cat"
    df[bin_name] = pd.cut(df[variable], bin_number, labels=labels)

def generate_dummy(df, variable):
    
    dummy = pd.get_dummies(df[variable])
    merged_df = pd.concat([df, dummy], axis=1)
    merged_df = merged_df.drop(columns=[variable])
    
    return merged_df

# Build Classifier

def split_and_create_X_y_set(df, test_size=0.3, rand=10):

    X_df = df.drop(labels=[config.PIPELINE_CONFIG['outcome_var']], axis=1)
    y_df = df[config.PIPELINE_CONFIG['outcome_var']]
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                        test_size=test_size,
                                                        random_state=rand)

    return X_train, X_test, y_train, y_test


# Evaluate Classifier

def evaluate_decision_tree_model(decision_tree, X_test, y_test):

    thold = config.PIPELINE_CONFIG['threshold']
    calc_thold = lambda x,y: 0 if x < y else 1
    pred_scores_test = decision_tree.predict_proba(X_test)[:,1]
    pred_test = np.array([calc_thold(sc, thold) for sc in pred_scores_test])
    test_acc = accuracy_score(pred_test, y_test)

    print("The calculated accuracy of this model is {:.4f}.".format(test_acc))


def find_best_max_depth(X_train, y_train, X_test, y_test):

    random_state = config.PIPELINE_CONFIG['random_state']

    for i in range(1, 10):
        dt = DTC(criterion='gini', max_depth=i, random_state=random_state)
        dt.fit(X_train, y_train)
        print('### max_depth: {}'.format(i))
        evaluate_decision_tree_model(dt, X_test, y_test)
        print('---------------')
