from backends import Pre_Data
from Conventional_Model import (Model1_DT, Model2_KNN, Model3_SVR, Model4_RF,
                                Model5_XGBoost, Model6_CatBoost, Model7_CNN, Model8_DNN)
from FE_Model import (Model9_FEIDEAL_XGB, Model10_FEIDEAL_DNN, Model11_FERF_XGB, Model12_FERF_DNN,
                      Model13_FEXGB_XGB, Model14_FEXGB_DNN, Model15_FEGMM_XGB, Model16_FEGMM_DNN)
import multiprocessing

def run_model_segments(model_func, data_config):
    pool = multiprocessing.Pool(processes=3)
    results = []
    for data_train, data_val, data_test, label in data_config:
        result = pool.apply_async(
            model_func,
            args=(data_train, data_val, data_test, label)
        )
        results.append(result)
    pool.close()
    pool.join()
    return [result.get() for result in results]

if __name__ == '__main__':
    # datasets
    predata = Pre_Data.Data_loading('1_backends/2_Materials_info.xlsx')
    data_train_Low, data_train_Medium, data_train_High = predata[0], predata[3], predata[6]
    data_val_Low, data_val_Medium, data_val_High = predata[1], predata[4], predata[7]
    data_test_Low, data_test_Medium, data_test_High = predata[2], predata[5], predata[8]

    data_configs = [
        (data_train_Low, data_val_Low, data_test_Low, 'Low'),
        (data_train_Medium, data_val_Medium, data_test_Medium, 'Medium'),
        (data_train_High, data_val_High, data_test_High, 'High'),
    ]

    #  Model1
    print("Running Model1 (DT)...")
    run_model_segments(Model1_DT.DT, data_configs)

    #  Model2
    print("Running Model2 (KNN)...")
    run_model_segments(Model2_KNN.KNN, data_configs)

    #  Model3
    print("Running Model3 (SVR)...")
    run_model_segments(Model3_SVR.ModelSVR, data_configs)

    #  Model4
    print("Running Model4 (RF)...")
    run_model_segments(Model4_RF.RF, data_configs)

    #  Model5
    print("Running Model5 (XGBoost)...")
    run_model_segments(Model5_XGBoost.XGBoost, data_configs)

    #  Model6
    print("Running Model6 (CatBoost)...")
    run_model_segments(Model6_CatBoost.CatBoost, data_configs)

    #  Model7
    print("Running Model7 (CNN)...")
    run_model_segments(Model7_CNN.CNN, data_configs)

    #  Model8
    print("Running Model8 (DNN)...")
    run_model_segments(Model8_DNN.DNN, data_configs)

    #  Model9
    print("Running Model9 (FEIDEAL_XGB)...")
    run_model_segments(Model9_FEIDEAL_XGB.FEIDEAL_XGB, data_configs)

    #  Model10
    print("Running Model10 (FEIDEAL_DNN)...")
    run_model_segments(Model10_FEIDEAL_DNN.FEIDEAL_DNN, data_configs)

    # Model11
    print("Running Model11 (FERF_XGB)...")
    run_model_segments(Model11_FERF_XGB.FERF_XGB, data_configs)

    # Model12
    print("Running Model12 (FERF_DNN)...")
    run_model_segments(Model12_FERF_DNN.FERF_XGB, data_configs)

    # Model13
    print("Running Model13 (FEXGB_XGB)...")
    run_model_segments(Model13_FEXGB_XGB.FEXGB_XGB, data_configs)

    # Model14
    print("Running Model14 (FEXGB_DNN)...")
    run_model_segments(Model14_FEXGB_DNN.FEXGB_DNN, data_configs)

    # Model15
    print("Running Model15 (FEGMM_XGB)...")
    run_model_segments(Model15_FEGMM_XGB.FEGMM_XGB, data_configs)

    # Model16
    print("Running Model16 (FEGMM_DNN)...")
    run_model_segments(Model16_FEGMM_DNN.FEGMM_DNN, data_configs)



    print("All models completed!")