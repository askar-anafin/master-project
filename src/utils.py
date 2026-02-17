import pandas as pd
import os
import ast
import numpy as np

def load_ptbxl_metadata(data_dir):
    """
    Load PTB-XL database CSV and mapping file, and add diagnostic superclasses.
    """
    csv_path = os.path.join(data_dir, 'ptbxl_database.csv')
    scp_path = os.path.join(data_dir, 'scp_statements.csv')
    
    df = pd.read_csv(csv_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    return df

def get_labels(df):
    """
    Convert diagnostic superclasses to one-hot or multi-label encoding.
    Classes: NORM, MI, STTC, CD, HYP
    """
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = np.zeros((len(df), len(classes)), dtype=np.float32)
    
    for i, (_, row) in enumerate(df.iterrows()):
        for label in row['diagnostic_superclass']:
            if label in classes:
                idx = classes.index(label)
                y[i, idx] = 1
                
    return y, classes
