# Import Packages
import pandas as pd
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def convert_to_iso(date_col):
    """
    Convert a date, originally in format YYYYMMDDHH,
    to ISO 8601 format (https://en.wikipedia.org/wiki/ISO_8601)
    
    Input: an array of DateTimes in YYYYMMDD format
    Output: an array of DateTimes in ISO 8601 format
    """
    date_col = date_col.astype(str)
    
    # year = YYYY
    year = date_col.str[0:4]
    # month = MM
    month = date_col.str[4:6]
    # day = DD
    day = date_col.str[6:8]
    # hour = HH
    hour = date_col.str[8:10]
    date_iso8601 = pd.to_datetime(year + '-' + month + '-' + day + 'T' + hour + ':00:00')
    
    return date_iso8601

def add_forecast_cat(wfn):
    """
    Add a forecast category column to the Wind Farm data
    Forecast Category 1:  1-12 hour forecasts
    Forecast Category 2: 13-24 hour forecasts
    Forecast Category 3: 25-36 hour forecasts
    Forecast Category 4: 37-48 hour forecasts
    
    Input: A DataFrame of Wind Farm data with column 'hors' containing hour-ahead forecasts 
    Output: The same DataFrame with an added column, 'forecast_cat' containing the forecast category
    """
    
    wfn['forecast_cat'] = None
    wfn.loc[ (wfn['hors'] >= 1) & (wfn['hors'] <= 12), 'forecast_cat'] = 1
    wfn.loc[ (wfn['hors'] >= 13) & (wfn['hors'] <= 24), 'forecast_cat'] = 2
    wfn.loc[ (wfn['hors'] >= 25) & (wfn['hors'] <= 36), 'forecast_cat'] = 3
    wfn.loc[ (wfn['hors'] >= 37) & (wfn['hors'] <= 48), 'forecast_cat'] = 4

    return wfn

def wfn_by_fc(wfn, forecast_cat):
    """
    Take a windfarm DataFrame and return a boolean sliced 
    version including data for a given forecast category
    
    Input: A DataFrame of Wind Farm data
    Output: The same DataFrame, but including only data for the requested forecast category
    """
    wfn = wfn.loc[(wfn['forecast_cat'] == forecast_cat)] # row slice
    return wfn

def lin_reg(wfn_fcn):
    """
    Run linear regression model on wind farm data:
    - Set y and X variables
    - Create training and test sets for y and X
    - Fit the regressor to the training data
    - Predict on the test data
    - Compute Root Mean Square Error to evaluate prediction accuracy
    
    Input: wfn_fcn, data for a specific wind farm and forecast category
    Output: Root Mean Square Error
    """
    reg=LinearRegression() # Create a linear regression object: reg
    # Initialize variables
    y = wfn_fcn[wp_lookup[key]]
    X = wfn_fcn.drop([wp_lookup[key], 'date', 'forecast_cat'], axis=1)
    # Create Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.3,
                                                        random_state=20)
    reg.fit(X_train, y_train) # Fit the regressor to the training data
    y_pred = reg.predict(X_test) # Predict on the test data        
    # Compute RMSE score and add to dictionary
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    linreg_scores[key]['fc' + str(i)] = rmse
    
def ridge_reg(wfn_fcn):
    """
    Run Ridge regression model on wind farm data:
    - Set y and X variables
    - Create training and test sets for y and X
    - Fit the regressor to the training data
    - Predict on the test data
    - Compute Root Mean Square Error to evaluate prediction accuracy
    
    Input: wfn_fcn, data for a specific wind farm and forecast category
    Output: Root Mean Square Error
    """
    ridge = Ridge(normalize=True)           # Create ridge regression object
    # Setup an array of alpha values to use for ridge regression
    alpha_space = np.logspace(-4, 0, 50)
    ridge_RMSEs  = []         # Setup list to store RMSE
    # Run ridge regression and compute RMSE over array of alphas
    #Initialize variables
    y = wfn_fcn[wp_lookup[key]]
    X = wfn_fcn.drop([wp_lookup[key], 'date', 'forecast_cat'], axis=1)
    # Create Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.3,
                                                        random_state=20)
    for alpha in alpha_space:
        ridge.alpha = alpha # Specify the alpha value to use: ridge.alpha
        ridge.fit(X_train, y_train) # Fit the Regressor to training data
        ridge_pred = ridge.predict(X_test)      # Predict on the test data
        # Calculate the RMSE, and append to ridge_RMSEs
        ridge_RMSEs.append(np.sqrt(mean_squared_error(y_test, ridge_pred)))
    
    # Retrieve best RMSE score and alpha value and add to dictionary
    rmse = min(ridge_RMSEs)
    alpha = alpha_space[np.argmin(ridge_RMSEs)]
    ridge_scores[key]['fc' + str(i)] = rmse

def lasso_reg(wfn_fcn):
    """
    Run Lasso regression model on wind farm data:
    - Set y and X variables
    - Create training and test sets for y and X
    - Fit the regressor to the training data
    - Predict on the test data
    - Compute Root Mean Square Error to evaluate prediction accuracy
    
    Input: wfn_fcn, data for a specific wind farm and forecast category
    Output: Root Mean Square Error
    """
    lasso = Lasso()           # Create lasso regression object
    lasso_RMSEs  = []         # Setup list to store RMSE
    # Setup an array of alpha values to use for lasso regression
    alpha_space = np.logspace(-4, 0, 50)
    # Run lasso regression and compute RMSE over array of alphas
    #Initialize variables
    y = wfn_fcn[wp_lookup[key]]
    X = wfn_fcn.drop([wp_lookup[key], 'date', 'forecast_cat'], axis=1)
    # Create Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.3,
                                                        random_state=20)
    for alpha in alpha_space:
        # Specify the alpha value to use: lasso.alpha
        lasso.alpha = alpha
        # Fit the Regressor to training data
        lasso.fit(X_train, y_train)             
        lasso_pred = lasso.predict(X_test)   # Predict on the test data
        # Calculate the RMSE, and append to lasso_RMSEs
        lasso_RMSEs.append(np.sqrt(mean_squared_error(y_test, lasso_pred)))
    
    # Retrieve best RMSE score and alpha value and add to dictionary
    rmse = min(lasso_RMSEs)
    alpha = alpha_space[np.argmin(lasso_RMSEs)]
    lasso_scores[key]['fc' + str(i)] = rmse

def reg_tree(wfn_fcn):
    """
    Run Decision Tree Regressor on wind farm data:
    - Set y and X variables
    - Create training and test sets for y and X
    - Fit the regressor to the training data
    - Predict on the test data
    - Compute Root Mean Square Error to evaluate prediction accuracy
    
    Input: wfn_fcn, data for a specific wind farm and forecast category
    Output: Root Mean Square Error
    """
    regtree = DecisionTreeRegressor(max_depth=5) # Create Tree regression object
    # Initialize variables
    y = wfn_fcn[wp_lookup[key]]
    X = wfn_fcn.drop([wp_lookup[key], 'date', 'forecast_cat'], axis=1)
    # Create Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.3,
                                                        random_state=20)
    regtree.fit(X_train, y_train) # Fit the regressor to the training data
    tree_pred = regtree.predict(X_test) # Predict on the test data        
    # Compute RMSE score and add to dictionary
    rmse = np.sqrt(mean_squared_error(y_test, tree_pred))
    regtree_scores[key]['fc' + str(i)] = rmse

def reg_nn(wfn_fcn):
    """
    Run Neural Network Regressor on wind farm data:
    - Set y and X variables
    - Create training and test sets for y and X
    - Fit the regressor to the training data
    - Predict on the test data
    - Compute Root Mean Square Error to evaluate prediction accuracy
    
    Input: wfn_fcn, data for a specific wind farm and forecast category
    Output: Root Mean Square Error
    """
    nn = MLPRegressor(hidden_layer_sizes=(10, )) # Create Neural Network Regression object
    # Initialize variables
    y = wfn_fcn[wp_lookup[key]]
    X = wfn_fcn.drop([wp_lookup[key], 'date', 'forecast_cat'], axis=1)
    # Create Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.3,
                                                        random_state=20)
    nn.fit(X_train, y_train) # Fit the regressor to the training data
    nn_pred = nn.predict(X_test) # Predict on the test data
    # Compute RMSE score and add to dictionary
    rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
    nn_scores[key]['fc' + str(i)] = rmse

if __name__ == '__main__':
    
    wf_dict = {'wf1': pd.read_csv('windforecasts_wf1.csv'),
               'wf2': pd.read_csv('windforecasts_wf2.csv'),
               'wf3': pd.read_csv('windforecasts_wf3.csv'),
               'wf4': pd.read_csv('windforecasts_wf4.csv'),
               'wf5': pd.read_csv('windforecasts_wf5.csv'),
               'wf6': pd.read_csv('windforecasts_wf6.csv'),
               'wf7': pd.read_csv('windforecasts_wf7.csv')}
    
    power = pd.read_csv('train.csv')
    power['date'] = convert_to_iso(power['date'])
    
    # Include only 2009-2010 data for wind power data
    power = power.loc[ (power['date'] >= '2009-07-01') & 
                       (power['date'] <=  '2010-12-31')]
    # Set index for wind power data
    power.set_index('date', inplace=True)   

    wp_lookup = {'wf1':'wp1',
                 'wf2':'wp2',
                 'wf3':'wp3',
                 'wf4':'wp4',
                 'wf5':'wp5',
                 'wf6':'wp6',
                 'wf7':'wp7'}

    scores = {'wf1':{'fc1':None, 'fc2':None, 'fc3':None, 'fc4':None},
              'wf2':{'fc1':None, 'fc2':None, 'fc3':None, 'fc4':None},
              'wf3':{'fc1':None, 'fc2':None, 'fc3':None, 'fc4':None},
              'wf4':{'fc1':None, 'fc2':None, 'fc3':None, 'fc4':None},
              'wf5':{'fc1':None, 'fc2':None, 'fc3':None, 'fc4':None},
              'wf6':{'fc1':None, 'fc2':None, 'fc3':None, 'fc4':None},
              'wf7':{'fc1':None, 'fc2':None, 'fc3':None, 'fc4':None}}
    
    linreg_scores = copy.deepcopy(scores)
    ridge_scores = copy.deepcopy(scores)
    lasso_scores = copy.deepcopy(scores)
    regtree_scores = copy.deepcopy(scores)
    nn_scores = copy.deepcopy(scores)
    
    for key, _ in wf_dict.items():
        
        # Convert date-times to ISO 8601 format
        wf_dict[key]['date'] = convert_to_iso(wf_dict[key]['date'])
        
        # Initialize mod_date column
        wf_dict[key]['mod_date'] = (wf_dict[key]['date'] + 
                                    pd.to_timedelta(arg=wf_dict[key]['hors'],unit='h'))
    
        # Initialize forecast_cat column
        wf_dict[key] = add_forecast_cat(wf_dict[key])
        
         # Include only 2009-2010 data for wind speed/direction data
        wf_dict[key] = wf_dict[key].loc[(wf_dict[key]['mod_date'] >= '2009-07-01') & 
                                        (wf_dict[key]['mod_date'] <= '2010-12-31')]
        
        # Set Index column
        wf_dict[key].set_index('mod_date',inplace=True)

        # Merge wind speed/direction data with wind power data
        wf_dict[key] = wf_dict[key].merge(power[[wp_lookup[key]]], 
                                          how='left',
                                          left_index=True,       
                                          right_index=True)
        
        for i in range(1,5):
        # Subset dataframe by forecast category
            wfn_fcn = wfn_by_fc(wfn = wf_dict[key], forecast_cat=i)
            
        # Apply Machine Learning Models
            lin_reg(wfn_fcn)   # Linear Regression
            ridge_reg(wfn_fcn) # Ridge Regression
            lasso_reg(wfn_fcn) # Lasso Regression
            reg_tree(wfn_fcn)  # Tree Regression
            reg_nn(wfn_fcn)    # Neural Networks
 
    linreg_scores = pd.DataFrame(pd.DataFrame(linreg_scores).transpose().stack(), columns=['linear_regression'])
    ridge_scores = pd.DataFrame(pd.DataFrame(ridge_scores).transpose().stack(), columns=['ridge'])
    lasso_scores = pd.DataFrame(pd.DataFrame(lasso_scores).transpose().stack(), columns=['lasso'])
    regtree_scores = pd.DataFrame(pd.DataFrame(regtree_scores).transpose().stack(), columns=['tree'])
    nn_scores = pd.DataFrame(pd.DataFrame(nn_scores).transpose().stack(), columns=['neural_network'])
    
    all_scores = linreg_scores.merge(ridge_scores, left_index=True, right_index=True)
    all_scores = all_scores.merge(lasso_scores, left_index=True, right_index=True)
    all_scores = all_scores.merge(regtree_scores, left_index=True, right_index=True)
    all_scores = all_scores.merge(nn_scores, left_index=True, right_index=True)

    # Plot results, one for each wind farm. 
    x = [1, 2, 3, 4] # x variable for graphing
    
    # List of Machine Learning Models for Visualization
    # Note: Linear regression, Ridge regression and Lasso produced very similar RMSE scores.
    # They are indistinguishable on the same plot with the Decision Tree and and Neural Network scores.
    # Given this, they are pooled together and displayed using the Linear Regression RMSE scores.
    models = ['linear_regression', 'tree', 'neural_network'] 
    
    for n in range(7):
        plt.figure()
        for i in models:
            plt.plot(x, all_scores[i][n*4:n*4+4], alpha=0.8, marker='.')
        plt.title('Machine Learning Results for Wind Farm ' + str(n+1))
        plt.xlabel('Forecast Category')
        plt.xticks([1, 2, 3, 4])
        plt.ylabel('Root Mean Square Error (RMSE)')
        plt.legend(['Linear Regression, Lasso, Ridge', 'Tree', 'Neural Network'])
