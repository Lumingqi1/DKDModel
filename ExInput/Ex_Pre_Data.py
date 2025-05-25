import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import RDKFingerprint, DataStructs
from rdkit.DataStructs.cDataStructs import CreateFromBitString
from sklearn.model_selection import train_test_split

# Generate the RDKit fingerprint dictionary
def RDKit_Featurize(filepath='1_compound_info.xlsx', fp_size=128, max_path=7):
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

def Ex_Data(path=None):
    # Load the master data
    if path == None:
        path = 'Ex_Data.xlsx'
    random_seed = 42
    df = pd.read_excel(path)
    selected_columns = [
        'Porous structure', 'Template', 'Si Precursor', 'Removal Template method',
        'Specific surface area', 'Pore volume', 'Pore size',
        'Amine content(N wt.%)', 'CO₂ pressure', 'Temperature', 'CO₂ capacity'
    ]
    df = df[selected_columns].copy()

    # Handling of missing values
    df = df.fillna(0)
    df.reset_index(drop=True, inplace=True)

    # Category feature processing
    df['Removal Template method'] = df['Removal Template method'].apply(lambda x: 1 if x == 'Retention' else 0)

    # 假设 df 是你的原始 DataFrame
    all_porous_categories = [
        'DOS', 'MSU', 'OMS', 'HMS', '3dd',
        'SBA-15', 'SBA-16', 'MCM-41', 'KIT-1', 'KIT-6'
    ]

    # 将 Porous structure 列转换为分类类型，并指定所有可能类别
    df['Porous structure'] = pd.Categorical(
        df['Porous structure'],
        categories=all_porous_categories
    )

    # One-Hot 编码，保持统一结构
    df = pd.get_dummies(df, columns=['Porous structure'], prefix='Porous')

    # 确保所有预期列都存在（即使当前没有数据）
    for porous in all_porous_categories:
        col = f'Porous_{porous}'
        if col not in df.columns:
            df[col] = 0

    # Obtain the RDKit fingerprint
    fp_dict = RDKit_Featurize(filepath='./backends/1_compound_info.xlsx')
    ref_fp = RDKFingerprint(Chem.MolFromSmiles("CCO"), fpSize=128, maxPath=7)

    # Replace with fingerprints and calculate the similarity
    for col in ['Template', 'Si Precursor']:
        df[col] = df[col].apply(lambda x: fp_dict.get(x, np.zeros(128, dtype=np.uint8)))
        df[col] = df[col].apply(lambda fp: calc_similarity(fp, ref_fp))

    return df
