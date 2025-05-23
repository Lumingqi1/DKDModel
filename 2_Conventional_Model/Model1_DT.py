from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from bayes_opt import BayesianOptimization
import numpy as np
import os
from joblib import dump
from sklearn.preprocessing import StandardScaler


class ModelOptimizer:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.best_model = None
        self.best_rmse = float('inf')
        self.best_cv_rmse = float('inf')
        self.best_params = {}

    def optimize_hyperparams(self, max_depth, min_samples_split, min_samples_leaf, max_features):
        params = {
            'max_depth': int(round(max_depth)),
            'min_samples_split': int(round(min_samples_split)),
            'min_samples_leaf': int(round(min_samples_leaf)),
            'max_features': max(0.1, min(1.0, max_features))
        }

        # Create and train Model1_DT
        model = DecisionTreeRegressor(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=42
        )

        model.fit(self.x_train, self.y_train)

        # Evaluate
        y_pred = model.predict(self.x_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))

        # Save the best model
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_params = params
            self.best_model = model

        return -rmse


def DT(train, val, test, type):
    # Data loading and preprocessing
    data_train = train
    data_val = val
    data_test = test

    x_train = data_train.drop(columns=['CO₂ capacity'])
    x_val = data_val.drop(columns=['CO₂ capacity'])
    x_test = data_test.drop(columns=['CO₂ capacity'])

    y_train = data_train['CO₂ capacity']
    y_val = data_val['CO₂ capacity']
    y_test = data_test['CO₂ capacity']

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Create optimizer
    optimizer = ModelOptimizer(x_train, y_train, x_val, y_val)

    pbounds = {
        'max_depth': (3, 20),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 20),
        'max_features': (0.1, 1.0)
    }

    # Run optimization
    bayes_optimizer = BayesianOptimization(
        f=optimizer.optimize_hyperparams,
        pbounds=pbounds,
        random_state=42
    )
    bayes_optimizer.maximize(init_points=20, n_iter=200)

    # Use the best model saved during optimization
    best_model = optimizer.best_model

    # Save best parameters
    optimizer_params = {f"opt1_{key}": val for key, val in optimizer.best_params.items()}

    os.makedirs('./output_params', exist_ok=True)
    pd.DataFrame([optimizer_params]).to_excel(
        f'./output_params/Model1_DT_{type}.xlsx',
        index=False
    )

    # Save model
    os.makedirs('./output_model', exist_ok=True)
    dump(best_model, f'./output_model/Model1_DT_{type}.joblib')

    # Save parameters and results
    y_pre_test = best_model.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pre_test))
    r2_test = r2_score(y_test, y_pre_test)

    y_pre_train = best_model.predict(x_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pre_train))
    r2_train = r2_score(y_train, y_pre_train)

    y_pre_val = best_model.predict(x_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pre_val))
    r2_val = r2_score(y_val, y_pre_val)

    result = {
        'name': 'DT',
        'rmse_train': rmse_train,
        'r2_train': r2_train,
        'rmse_val': rmse_val,
        'r2_val': r2_val,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'type': type
    }

    os.makedirs('./output_results', exist_ok=True)
    summary_path = './output_results/Summary.xlsx'
    if not os.path.exists(summary_path):
        pd.DataFrame([result]).to_excel(summary_path, index=False)
    else:
        existing_data = pd.read_excel(summary_path)
        updated_data = pd.concat([existing_data, pd.DataFrame([result])], ignore_index=True)
        updated_data.to_excel(summary_path, index=False)
