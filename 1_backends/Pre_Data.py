import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import RDKFingerprint, DataStructs
from rdkit.DataStructs.cDataStructs import CreateFromBitString
from sklearn.model_selection import train_test_split

# Generate the RDKit fingerprint dictionary
def RDKit_Featurize(filepath='./1_backends/1_compound_info.xlsx', fp_size=128, max_path=7):
    data = pd.read_excel(filepath)
    fp_dict = {}

    for _, row in data.iterrows():
        smiles = row.get('SMILES')
        abbrev = row.get('Abbreviation')
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = RDKFingerprint(mol, fpSize=fp_size, maxPath=max_path)
            fp_dict[abbrev] = np.array(fp, dtype=np.uint8)
        else:
            fp_dict[abbrev] = np.zeros(fp_size, dtype=np.uint8)

    return fp_dict


# Calculate the similarity with the reference molecule (EtOH(CCO))
def calc_similarity(fp_array, ref_fp):
    if isinstance(fp_array, np.ndarray):
        try:
            bitstring = ''.join(map(str, fp_array.tolist()))
            bitvect = CreateFromBitString(bitstring)
            return DataStructs.TanimotoSimilarity(ref_fp, bitvect)
        except:
            return 0.0
    return 0.0


def Data_loading(path):
    # Load the master data
    random_seed = 42
    df = pd.read_excel(path)
    selected_columns = [
        'Porous structure', 'Template', 'Si Precursor', 'Removal Template method',
        'Specific surface area', 'Pore volume', 'Pore size',
        'Amine content(N wt.%)', 'CO₂ pressure', 'Temperature', 'CO₂ capacity'
    ]
    df = df[selected_columns].copy()

    # Handling of missing values
    df[['Template', 'Si Precursor']] = df[['Template', 'Si Precursor']].fillna('0')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Category feature processing
    df['Removal Template method'] = df['Removal Template method'].apply(lambda x: 1 if x == 'Retention' else 0)

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Porous structure'], prefix='Porous')

    # Obtain the RDKit fingerprint
    fp_dict = RDKit_Featurize(filepath='./1_backends/1_compound_info.xlsx')
    ref_fp = RDKFingerprint(Chem.MolFromSmiles("CCO"), fpSize=128, maxPath=7)

    # Replace with fingerprints and calculate the similarity
    for col in ['Template', 'Si Precursor']:
        df[col] = df[col].apply(lambda x: fp_dict.get(x, np.zeros(128, dtype=np.uint8)))
        df[col] = df[col].apply(lambda fp: calc_similarity(fp, ref_fp))

    # Split the training, validation and test sets
    df1 = df[[
        'Porous_DOS',
        'Porous_MSU',
        'Porous_OMS',
        'Porous_HMS',
        'Porous_3dd',
        'Porous_SBA-15',
        'Porous_SBA-16',
        'Porous_MCM-41',
        'Porous_KIT-1',
        'Porous_KIT-6',
        'Template',
        'Si Precursor',
        'Removal Template method',
        'Specific surface area',
        'Pore volume',
        'Pore size',
        'Amine content(N wt.%)',
        'CO₂ pressure',
        'Temperature',
        'CO₂ capacity']]
    Train_Dataset_High, temp = train_test_split(df1, test_size=0.3, random_state=random_seed)
    Val_Dataset_High, Test_Dataset_High = train_test_split(temp, test_size=1 / 3, random_state=random_seed)

    # Split the training, validation and test sets
    df2 = df[['Removal Template method',
              'Specific surface area',
              'Pore volume',
              'Pore size',
              'Amine content(N wt.%)',
              'CO₂ pressure',
              'Temperature',
              'CO₂ capacity']]
    Train_Dataset_Medium, temp = train_test_split(df2, test_size=0.3, random_state=random_seed)
    Val_Dataset_Medium, Test_Dataset_Midium = train_test_split(temp, test_size=1 / 3, random_state=random_seed)

    # Split the training, validation and test sets
    df3 = df[['Specific surface area',
              'Pore volume',
              'Pore size',
              'Amine content(N wt.%)',
              'CO₂ pressure',
              'Temperature',
              'CO₂ capacity']]
    Train_Dataset_Low, temp = train_test_split(df3, test_size=0.3, random_state=random_seed)
    Val_Dataset_Low, Test_Dataset_Low = train_test_split(temp, test_size=1 / 3, random_state=random_seed)

    return (Train_Dataset_Low, Val_Dataset_Low, Test_Dataset_Low,
            Train_Dataset_Medium, Val_Dataset_Medium, Test_Dataset_Midium,
            Train_Dataset_High, Val_Dataset_High, Test_Dataset_High)