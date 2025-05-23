from ExInput import Ex_Pre_Data
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from backends import Pre_Data
from sklearn.mixture import GaussianMixture


def save_tools(y_true, y_pred, model_name, type):

    rmse_test = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_test = r2_score(y_true, y_pred)

    pre = pd.DataFrame({
        'true_val': y_true,
        'pred_val': y_pred
    })

    output_dir = 'output_ex_results'
    os.makedirs(output_dir, exist_ok=True)

    pre.to_excel(
        os.path.join(output_dir, f'{model_name}_{type}.xlsx'),
        index=False
    )

    result = {
        'Name': model_name,
        'Dataset': type,
        'rmse_test': rmse_test,
        'r2_test': r2_test
    }

    summary_path = os.path.join(output_dir, 'Summary.xlsx')
    if not os.path.exists(summary_path):
        pd.DataFrame([result]).to_excel(summary_path, index=False)
    else:
        existing_data = pd.read_excel(summary_path)
        updated_data = pd.concat(
            [existing_data, pd.DataFrame([result])],
            ignore_index=True
        )
        updated_data.to_excel(summary_path, index=False)

    return

if __name__ == '__main__':
    random_seed = 42

    predata = Pre_Data.Data_loading('1_backends/2_Materials_info.xlsx')

    # datasets
    data_train_Low, data_train_Medium, data_train_High = predata[0], predata[3], predata[6]
    data_val_Low, data_val_Medium, data_val_High = predata[1], predata[4], predata[7]
    data_test_Low, data_test_Medium, data_test_High = predata[2], predata[5], predata[8]

    for type in ['Low','Medium','High']:
        if type == 'Low':
            params = ['Specific surface area', 'Pore volume', 'Pore size',
                      'Amine content(N wt.%)', 'CO₂ pressure', 'Temperature', 'CO₂ capacity']
            data_train = data_train_Low
        elif type == 'Medium':
            params = ['Removal Template method', 'Specific surface area', 'Pore volume', 'Pore size',
                      'Amine content(N wt.%)', 'CO₂ pressure', 'Temperature', 'CO₂ capacity']
            data_train = data_train_Medium
        elif type == 'High':
            params = ['Porous_DOS', 'Porous_MSU', 'Porous_OMS', 'Porous_HMS', 'Porous_3dd',
                      'Porous_SBA-15', 'Porous_SBA-16', 'Porous_MCM-41', 'Porous_KIT-1', 'Porous_KIT-6',
                      'Template', 'Si Precursor', 'Removal Template method', 'Specific surface area',
                      'Pore volume', 'Pore size', 'Amine content(N wt.%)', 'CO₂ pressure',
                      'Temperature', 'CO₂ capacity']
            data_train = data_train_High

        data = Ex_Pre_Data.Ex_Data('4_ExInput/Ex_Data.xlsx')
        data = data[params].copy()

        x_train = data_train.drop(columns=['CO₂ capacity'])
        x_test = data.drop(columns=['CO₂ capacity'])
        y_test = data['CO₂ capacity']

        scaler = StandardScaler()
        x_train_scaler = scaler.fit_transform(x_train)
        x_test_scaler = scaler.transform(x_test)

        model_list = [
            f'./output_model/Model1_DT_{type}.joblib',
            f'./output_model/Model2_KNN_{type}.joblib',
            f'./output_model/Model3_SVR_{type}.joblib',
            f'./output_model/Model4_RF_{type}.joblib',
            f'./output_model/Model5_XGBoost_{type}.joblib',
            f'./output_model/Model6_CatBoost_{type}.joblib',
            f'./output_model/Model7_CNN_{type}.h5',
            f'./output_model/Model8_DNN_{type}.h5'
        ]

        for model_path in model_list:
            if model_path.endswith('.h5'):
                x_test_scaler = x_test_scaler.reshape(-1, x_test_scaler.shape[1], 1)
                best_model = load_model(model_path)
                model_name = os.path.basename(model_path).replace('.h5', '')
                y_pre_test = best_model.predict(x_test_scaler)[:,0]

            else:
                best_model = load(model_path)
                model_name = os.path.basename(model_path).replace('.joblib', '')
                y_pre_test = best_model.predict(x_test_scaler)

            save_tools(y_test, y_pre_test, model_name, type)


        # model11
        best_model = load(f'./output_model/Model11_RF_{type}.joblib')
        y_class_train_pred = best_model.predict(x_train)
        y_class_test_pred = best_model.predict(x_test)

        y_class_train = x_train.copy()
        y_class_test = x_test.copy()

        y_class_train['y_class'] = y_class_train_pred
        y_class_test['y_class'] = y_class_test_pred

        scaler = StandardScaler()
        x_train_scaler = scaler.fit_transform(y_class_train)
        x_test_scaler = scaler.transform(y_class_test)

        model_path=f'./output_model/Model11_FERF_XGB_{type}.joblib'
        best_model2 = load(model_path)
        model_name = os.path.basename(model_path).replace('.joblib', '')

        y_pre_test = best_model2.predict(x_test_scaler)
        save_tools(y_test, y_pre_test, model_name, type)

        # model12
        best_model = load(f'./output_model/Model12_RF_{type}.joblib')
        y_class_train_pred = best_model.predict(x_train)
        y_class_test_pred = best_model.predict(x_test)
        y_class_train = x_train.copy()
        y_class_test = x_test.copy()

        y_class_train['y_class'] = y_class_train_pred
        y_class_test['y_class'] = y_class_test_pred

        scaler = StandardScaler()
        x_train_scaler = scaler.fit_transform(y_class_train)
        x_test_scaler = scaler.transform(y_class_test)

        x_train_scaler = x_train_scaler.reshape(-1, x_train_scaler.shape[1], 1)
        x_test_scaler = x_test_scaler.reshape(-1, x_test_scaler.shape[1], 1)

        model_path=f'./output_model/Model12_FERF_DNN_{type}.h5'
        best_model2 = load_model(model_path)
        model_name = os.path.basename(model_path).replace('.h5', '')
        y_pre_test = best_model2.predict(x_test_scaler)[:,0]
        save_tools(y_test, y_pre_test, model_name, type)

        # model13
        best_model = load(f'./output_model/Model13_XGB_{type}.joblib')
        y_class_train = x_train.copy()
        y_class_test = x_test.copy()

        scaler = StandardScaler()
        x_train_scaler = scaler.fit_transform(x_train)
        x_test_scaler = scaler.transform(x_test)

        y_class_train['y_class'] = best_model.predict(x_train_scaler) / x_train['Amine content(N wt.%)']
        y_class_test['y_class'] = best_model.predict(x_test_scaler) / x_test['Amine content(N wt.%)']

        scaler = StandardScaler()
        x_train_scaler = scaler.fit_transform(y_class_train)
        x_test_scaler = scaler.transform(y_class_test)

        model_path=f'./output_model/Model13_FEXGB_XGB_{type}.joblib'
        best_model2 = load(model_path)
        model_name = os.path.basename(model_path).replace('.joblib', '')

        y_pre_test = best_model2.predict(x_test_scaler)
        save_tools(y_test, y_pre_test, model_name, type)

        # model14
        best_model = load(f'./output_model/Model14_XGB_{type}.joblib')
        y_class_train = x_train.copy()
        y_class_test = x_test.copy()

        scaler = StandardScaler()
        x_train_scaler = scaler.fit_transform(x_train)
        x_test_scaler = scaler.transform(x_test)

        y_class_train['y_class'] = best_model.predict(x_train_scaler) / x_train['Amine content(N wt.%)']
        y_class_test['y_class'] = best_model.predict(x_test_scaler) / x_test['Amine content(N wt.%)']

        scaler = StandardScaler()
        x_train_scaler = scaler.fit_transform(y_class_train)
        x_test_scaler = scaler.transform(y_class_test)

        x_train_scaler = x_train_scaler.reshape(-1, x_train_scaler.shape[1], 1)
        x_test_scaler = x_test_scaler.reshape(-1, x_test_scaler.shape[1], 1)

        model_path = f'./output_model/Model14_FEXGB_DNN_{type}.h5'
        best_model2 = load_model(model_path)
        model_name = os.path.basename(model_path).replace('.h5', '')
        y_pre_test = best_model2.predict(x_test_scaler)[:,0]

        save_tools(y_test, y_pre_test, model_name, type)

        # model15
        y_class_train = x_train.copy()
        y_class_test = x_test.copy()

        gmm = GaussianMixture(n_components=3).fit(x_train)

        y_class_train['y_class'] = gmm.predict(x_train)
        y_class_test['y_class'] = gmm.predict(x_test)

        scaler = StandardScaler()
        x_train_scaler = scaler.fit_transform(y_class_train)
        x_test_scaler = scaler.transform(y_class_test)

        model_path = f'./output_model/Model15_FEGMM_XGB_{type}.joblib'
        best_model2 = load(model_path)
        model_name = os.path.basename(model_path).replace('.joblib', '')

        y_pre_test = best_model2.predict(x_test_scaler)
        save_tools(y_test, y_pre_test, model_name, type)

        # model16
        y_class_train = x_train.copy()
        y_class_test = x_test.copy()

        gmm = GaussianMixture(n_components=3).fit(x_train)

        y_class_train['y_class'] = gmm.predict(x_train)
        y_class_test['y_class'] = gmm.predict(x_test)

        scaler = StandardScaler()
        x_train_scaler = scaler.fit_transform(y_class_train)
        x_test_scaler = scaler.transform(y_class_test)

        x_train_scaler = x_train_scaler.reshape(-1, x_train_scaler.shape[1], 1)
        x_test_scaler = x_test_scaler.reshape(-1, x_test_scaler.shape[1], 1)

        model_path = f'./output_model/Model16_FEGMM_DNN_{type}.h5'
        best_model2 = load_model(model_path)
        model_name = os.path.basename(model_path).replace('.h5', '')
        y_pre_test = best_model2.predict(x_test_scaler)[:,0]

        save_tools(y_test, y_pre_test, model_name, type)