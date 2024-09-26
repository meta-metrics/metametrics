import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import json
import logging

from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.metrics import make_scorer

logging.basicConfig(level=logging.INFO)

SPLIT = 0.7
RANDOM_STATE_DATA = 1

def prepare_data(ori_df, group_system):
    train_df, test_df = train_test_split(ori_df, test_size=SPLIT, random_state=RANDOM_STATE_DATA, stratify=ori_df[group_system])
    return train_df, test_df

# Define the custom Kendall scorer
def kendall_scorer(y_true, y_pred):
    # Calculate Kendall's Tau correlation
    score = stats.kendalltau(y_true, y_pred).correlation
    if np.isnan(score):
        return -1
    return score

def optimize_xgb(xgb_json_seg_path, train_df, human_score_col_name, metrics_used):
    X_train = train_df[metrics_used]
    Y_train = train_df[human_score_col_name].to_list()
    
    with open(xgb_json_seg_path, 'r') as file:
        json_content = json.load(file)

    fixed_param = json_content.get("fixed_param", {})

    # Initialize BayesSearchCV through parameter space
    param_space = {}
    for param_name, param_info in json_content["param_space"].items():
        param_type = param_info["type"]
        low = param_info["low"]
        high = param_info["high"]
        prior = param_info["prior"] if "prior" in param_info else "uniform"
        if param_type == "Integer":
            param_space[param_name] = Integer(low, high, prior=prior)
        elif param_type == "Real":
            param_space[param_name] = Real(low, high, prior=prior)
        elif param_type == "Categorical":
            step = param_info["step"] if "step" in param_info else 1
            param_space[param_name] = Categorical(list(range(low, high + 1, step)))
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    n_iter = json_content["n_iter"]

    # Create the scorer function
    scorer = make_scorer(kendall_scorer, greater_is_better=True)

    # Initialize the XGBoost model
    xgb_model = XGBRegressor(tree_method='auto', objective="reg:absoluteerror", random_state=RANDOM_STATE_DATA, **fixed_param)

    logging.info('Running XGB Bayes Search for Seg')
    logging.info(f"Parameter space for Seg: {param_space}")
    hpm_search_algorithm = BayesSearchCV(estimator=xgb_model,
                                            search_spaces=param_space,
                                            scoring=scorer,
                                            cv=10,
                                            n_iter=n_iter,
                                            random_state=RANDOM_STATE_DATA,
                                            n_jobs=-1)

    # Fit the grid search to the data
    hpm_search_algorithm.fit(X_train, Y_train)
    logging.info("Best params for seg:")
    logging.info(hpm_search_algorithm.best_params_)

    # Get the best model
    best_model = hpm_search_algorithm.best_estimator_
    logging.info("Feature importance for seg:")
    logging.info(best_model.feature_importances_)
    
    logging.info(f"Segment kendall corr after training: {kendall_scorer(hpm_search_algorithm.predict(X_train), Y_train)}")

    return best_model

def get_optimized_models(ori_df, metrics_used, human_score_col_name, group_system, xgb_json_path_seg):
    """
    Optimize models using the provided data and metrics for segment-level and system-level performance.

    Parameters:
    ----------
    ori_df : pandas.DataFrame
        The original DataFrame containing metric scores along with human evaluation scores.
        
    metrics_used : list of str
        A list of metric column names used for model optimization. These metrics will be used 
        as features in the model training process.
        
    human_score_col_name : str
        The column name in `ori_df` that contains the human evaluation scores. This is the 
        target variable the models will be optimized against.
        
    group_system : list of str
        A list of column names used to group for system evaluations within the data.

    xgb_json_path_seg : str
        Path to the JSON configuration file for the XGBoost model, which contains the 
        hyperparameters and settings used for training the segment-level model.
    """
    train_df, test_df = prepare_data(ori_df, group_system)
    seg_model = optimize_xgb(xgb_json_path_seg, train_df, human_score_col_name, metrics_used)
    
    # Evaluation now... if want to skip just go to return
    full_df = ori_df.copy(True)

    # Do evaluation based on segment and system (overall)
    full_df["metametrics_score"] = seg_model.predict(full_df[metrics_used])
    system_df = full_df.copy(True).groupby(group_system).agg({'metametrics_score': 'mean', human_score_col_name: 'mean'}).reset_index()
    kt = stats.kendalltau(full_df['metametrics_score'], full_df[human_score_col_name], nan_policy='omit')
    logging.info(f"Overall segment-level score for {human_score_col_name} is {kt.correlation}")
    
    kt = stats.kendalltau(system_df['metametrics_score'], system_df[human_score_col_name], nan_policy='omit')
    logging.info(f"Overall system-level score for {human_score_col_name} is {kt.correlation}")
    
    # Do evaluation based on segment and system (test)
    train_df, test_df = prepare_data(full_df, group_system)
    system_test_df = test_df.copy(True).groupby(group_system).agg({'metametrics_score': 'mean', human_score_col_name: 'mean'}).reset_index()

    kt = stats.kendalltau(test_df['metametrics_score'], test_df[human_score_col_name], nan_policy='omit')
    logging.info(f"Test segment-level score for {human_score_col_name} is {kt.correlation}")
    
    kt = stats.kendalltau(system_test_df['metametrics_score'], system_test_df[human_score_col_name], nan_policy='omit')
    logging.info(f"Test system-level score for {human_score_col_name} is {kt.correlation}")
            
    return seg_model
