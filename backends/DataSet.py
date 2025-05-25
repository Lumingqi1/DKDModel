import pandas as pd

def DataSet(Path_File=None, Sheet_Name = None):
    if Path_File == None:
        Path_File = "../2025_ML_CC/backends/data.xlsx"

    if Sheet_Name == None:
        Sheet_Name = "Dataset"

    df = pd.read_excel(Path_File)

    return df
