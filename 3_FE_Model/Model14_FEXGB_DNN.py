from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from bayes_opt import BayesianOptimization
from joblib import load, dump
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import  Dense
from tensorflow.keras.models import Sequential
import xgboost as xgb

def create_DNN_model(input_shape, dense_units1,dense_units2, dense_units3, dense_units4):
    model = Sequential()
    model.add(Dense(units=dense_units1, input_dim=input_shape, activation='relu'))
    model.add(Dense(units=dense_units2, activation='relu'))
    model.add(Dense(units=dense_units3, activation='relu'))
    model.add(Dense(units=dense_units4, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


class ModelOptimizer:
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

        # Create and train XGB
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

class ModelOptimizer2:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.best_model = None
        self.best_rmse = float('inf')
        self.best_params = {}

    def optimize_hyperparams(self, units1, units2, units3, units4, epochs, batch_size):
        params = {
            'units1': int(round(units1)),
            'units2': int(round(units2)),
            'units3': int(round(units3)),
            'units4': int(round(units4)),
            'epochs': int(round(epochs)),
            'batch_size': int(round(batch_size))
        }

        # Create and train Model8_DNN
        model = create_DNN_model(
            self.x_train.shape[1],
            params['units1'],
            params['units2'],
            params['units3'],
            params['units4']
        )

        model.fit(
            self.x_train,
            self.y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=0
        )

        # Evaluate
        y_pred = model.predict(self.x_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))

        # Save the best model
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_params = params
            self.best_model = model

        return -rmse

def FEXGB_DNN(train, val, test, type):
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
    x_train_scaler = scaler.fit_transform(x_train)
    x_val_scaler = scaler.transform(x_val)
    x_test_scaler = scaler.transform(x_test)

    # # Create optimizer
    # optimizer = ModelOptimizer(x_train_scaler, y_train, x_val_scaler, y_val)
    #
    # pbounds = {
    #     'n_estimators': (100, 500),
    #     'max_depth': (5, 30),
    # }
    #
    # # Run optimization
    # bayes_optimizer = BayesianOptimization(
    #     f=optimizer.optimize_hyperparams,
    #     pbounds=pbounds,
    #     random_state=42
    # )
    # bayes_optimizer.maximize(init_points=20, n_iter=200)

    # # Use the best model saved during optimization
    # best_model = optimizer.best_model
    #
    # # Save model
    # os.makedirs('./output_model', exist_ok=True)
    # dump(best_model, f'./output_model/Model14_XGB_{type}.joblib')

    best_model = load(f'./output_model/Model14_XGB_{type}.joblib')

    scaler = StandardScaler()
    x_train_scaler = scaler.fit_transform(x_train)
    x_val_scaler = scaler.transform(x_val)
    x_test_scaler = scaler.transform(x_test)

    x_train['y_class'] = best_model.predict(x_train_scaler) / x_train['Amine content(N wt.%)']
    x_val['y_class'] = best_model.predict(x_val_scaler) / x_val['Amine content(N wt.%)']
    x_test['y_class'] = best_model.predict(x_test_scaler) / x_test['Amine content(N wt.%)']

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    x_train = x_train.reshape(-1, x_train.shape[1], 1)
    x_val = x_val.reshape(-1, x_val.shape[1], 1)
    x_test = x_test.reshape(-1, x_test.shape[1], 1)

    # Create optimizer2
    optimizer2 = ModelOptimizer2(x_train, y_train, x_val, y_val)

    pbounds2 = {
        'units1': (166, 166),
        'units2': (64, 64),
        'units3': (209, 209),
        'units4': (122, 122),
        'epochs': (50, 200),
        'batch_size': (4, 32)
    }

    # Run optimization2
    bayes_optimizer2 = BayesianOptimization(
        f=optimizer2.optimize_hyperparams,
        pbounds=pbounds2,
        random_state=42
    )
    bayes_optimizer2.maximize(init_points=10, n_iter=100)

    # Use the best model saved during optimization
    best_model2 = optimizer2.best_model

    # Save best parameters
    # optimizer_params1 = {f"opt1_{key}": val for key, val in optimizer.best_params.items()}
    optimizer_params= {f"opt1_{key}": val for key, val in optimizer2.best_params.items()}
    # optimizer_params = {k: v for d in [optimizer_params1, optimizer_params2] for k, v in d.items()}

    os.makedirs('./output_params', exist_ok=True)
    pd.DataFrame([optimizer_params]).to_excel(
        f'./output_params/Model14_FEXGB_DNN_{type}.xlsx',
        index=False
    )

    # Save model
    os.makedirs('./output_model', exist_ok=True)
    best_model2.save(f'./output_model/Model14_FEXGB_DNN_{type}.h5')

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
        'name': 'FEXGB_DNN',
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