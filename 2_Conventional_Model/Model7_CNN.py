import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization

# Create Model7_CNN
def create_CNN_model(input_shape, filters, dense_units1, dense_units2):
    model = Sequential([
        Conv1D(filters=filters, kernel_size=3, activation='relu',
               input_shape=(input_shape, 1)),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dense(dense_units1, activation='relu'),
        Dense(dense_units2, activation='relu'),
        Dense(dense_units2, activation='relu'),
        Dropout(rate=0.3),  # 修正参数名
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


class ModelOptimizer:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train.values.reshape(-1, 1)
        self.x_val = x_val
        self.y_val = y_val.values.reshape(-1, 1)
        self.best_model = None
        self.best_rmse = float('inf')
        self.best_params = {}

    def optimize_hyperparams(self, filters, dense_units1, dense_units2, epochs, batch_size):
        params = {
            'filters': int(round(filters)),
            'dense_units1': int(round(dense_units1)),
            'dense_units2': int(round(dense_units2)),
            'epochs': int(round(epochs)),
            'batch_size': int(round(batch_size))
        }

        # Create and train Model8_DNN
        model = create_CNN_model(
            input_shape=self.x_train.shape[1],
            filters=params['filters'],
            dense_units1=params['dense_units1'],
            dense_units2=params['dense_units2']
        )

        model.fit(
            self.x_train, self.y_train,
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


def CNN(train, val, test, type):
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

    x_train = x_train.reshape(-1, x_train.shape[1], 1)
    x_val = x_val.reshape(-1, x_val.shape[1], 1)
    x_test = x_test.reshape(-1, x_test.shape[1], 1)

    # Create optimizer
    optimizer = ModelOptimizer(x_train, y_train, x_val, y_val)

    pbounds = {
        'filters': (16, 256),
        'dense_units1': (16, 256),
        'dense_units2': (16, 256),
        'epochs': (50, 200),
        'batch_size': (4, 32)
    }

    # Run optimization
    bayes_opt = BayesianOptimization(
        f=optimizer.optimize_hyperparams,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )
    bayes_opt.maximize(init_points=20, n_iter=200)

    # Use the best model saved during optimization
    best_model = optimizer.best_model

    # Save best parameters
    optimizer_params = {f"opt1_{key}": val for key, val in optimizer.best_params.items()}

    os.makedirs('./output_params', exist_ok=True)
    pd.DataFrame([optimizer_params]).to_excel(
        f'./output_params/Model7_CNN_{type}.xlsx',
        index=False
    )

    # Save model
    os.makedirs('./output_model', exist_ok=True)
    best_model.save(f'./output_model/Model7_CNN_{type}.h5')

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
        'name': 'CNN',
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