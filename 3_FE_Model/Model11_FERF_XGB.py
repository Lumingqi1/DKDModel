from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pandas as pd
from bayes_opt import BayesianOptimization
from joblib import dump, load
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class ModelOptimizer:
    def __init__(self, x_train, y_class_train, x_verification, y_class_verification):
        self.x_train = x_train
        self.y_class_train = y_class_train
        self.x_verification = x_verification
        self.y_class_verification = y_class_verification
        self.best_model = None
        self.best_accuracy = 0
        self.best_params = {}

    def optimize_hyperparams(self, n_estimators, max_depth, min_samples_split):
        params = {
            'n_estimators': int(round(n_estimators)),
            'max_depth': int(round(max_depth)),
            'min_samples_split': int(round(min_samples_split)),
        }

        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        model.fit(self.x_train, self.y_class_train)

        y_class_verification_p = model.predict(self.x_verification)
        accuracy = accuracy_score(self.y_class_verification, y_class_verification_p)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_params = params
            self.best_model = model

        return accuracy

class ModelOptimizer2:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.best_model = None
        self.best_rmse = float('inf')
        self.best_params = {}

    def optimize_hyperparams(self, n_estimators, max_depth):
        params = {
            'n_estimators': int(round(n_estimators)),
            'max_depth': int(round(max_depth)),
        }

        # Create and train model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',  # 回归任务
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
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

def FERF_XGB(train, val, test, type):
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

    bins = [0, 0.16, 0.32, 1]
    labels = [0, 1, 2]
    data_train['N_efficiency'] = pd.cut(
        data_train['CO₂ capacity'] / data_train['Amine content(N wt.%)'],
        bins=bins, labels=labels, right=False
    )
    data_val['N_efficiency'] = pd.cut(
        data_val['CO₂ capacity'] / data_val['Amine content(N wt.%)'],
        bins=bins, labels=labels, right=False
    )
    data_test['N_efficiency'] = pd.cut(
        data_test['CO₂ capacity'] / data_test['Amine content(N wt.%)'],
        bins=bins, labels=labels, right=False
    )
    y_class_train = data_train['N_efficiency']
    y_class_val = data_val['N_efficiency']
    y_class_test = data_test['N_efficiency']

    # Create optimizer
    optimizer = ModelOptimizer(x_train, y_class_train, x_val, y_class_val)

    pbounds = {
        'n_estimators': (100, 500),
        'max_depth': (5, 30),
        'min_samples_split': (2, 10)
    }

    # Run optimization
    bayes_optimizer = BayesianOptimization(
        f=optimizer.optimize_hyperparams,
        pbounds=pbounds,
        random_state=42
    )
    bayes_optimizer.maximize(init_points=20, n_iter=200)

    best_model = optimizer.best_model

    os.makedirs('./output_model', exist_ok=True)
    dump(best_model, f'./output_model/Model11_RF_{type}.joblib')

    y_class_train_pred = best_model.predict(x_train)
    y_class_val_pred = best_model.predict(x_val)
    y_class_test_pred = best_model.predict(x_test)

    x_train['y_class'] = y_class_train_pred
    x_val['y_class'] = y_class_val_pred
    x_test['y_class'] = y_class_test_pred

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Create optimizer2
    optimizer2 = ModelOptimizer2(x_train, y_train, x_val, y_val)

    pbounds = {
        'n_estimators': (100, 500),
        'max_depth': (5, 30),
    }

    # Run optimization2
    bayes_optimizer2 = BayesianOptimization(
        f=optimizer2.optimize_hyperparams,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )
    bayes_optimizer2.maximize(init_points=15, n_iter=200)

    # Use the best model
    best_model2 = optimizer2.best_model

    # Save best parameters
    optimizer_params1 = {f"opt1_{key}": val for key, val in optimizer.best_params.items()}
    optimizer_params2 = {f"opt1_{key}": val for key, val in optimizer2.best_params.items()}
    optimizer_params = {k: v for d in [optimizer_params1, optimizer_params2] for k, v in d.items()}

    os.makedirs('./output_params', exist_ok=True)
    pd.DataFrame([optimizer_params]).to_excel(
        f'./output_params/Model11_FERF_XGB_{type}.xlsx',
        index=False
    )

    # Save model
    os.makedirs('./output_model', exist_ok=True)
    dump(best_model2, f'./output_model/Model11_FERF_XGB_{type}.joblib')

    # Save parameters and results
    y_pre_test = best_model2.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pre_test))
    r2_test = r2_score(y_test, y_pre_test)

    y_pre_train = best_model2.predict(x_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pre_train))
    r2_train = r2_score(y_train, y_pre_train)

    y_pre_val = best_model2.predict(x_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pre_val))
    r2_val = r2_score(y_val, y_pre_val)

    result = {
        'name': 'FERF_XGB',
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