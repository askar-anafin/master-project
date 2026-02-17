import pandas as pd
import os
import ast

def analyze_ptb_xl_classes():
    print("\n--- PTB-XL Class Analysis ---")
    data_path = os.path.join('data', 'raw', 'ptb-xl')
    csv_path = os.path.join(data_path, 'ptbxl_database.csv')
    scp_path = os.path.join(data_path, 'scp_statements.csv')
    
    if not os.path.exists(csv_path):
        print("ptbxl_database.csv not found.")
        return

    try:
        # Load database index
        df = pd.read_csv(csv_path, index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load SCP statements for aggregation
        agg_df = pd.read_csv(scp_path, index_col=0)
        
        # We need to map diagnostic codes to superclasses
        # Filter for diagnostic statements
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # Apply aggregation
        df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
        
        # Flatten the list to count classes
        all_classes = [item for sublist in df.diagnostic_superclass for item in sublist]
        
        print(f"Total Records: {len(df)}")
        print("\nDiagnostic Superclass Distribution:")
        print(pd.Series(all_classes).value_counts())
        
    except Exception as e:
        print(f"Error analyzing PTB-XL: {e}")

if __name__ == "__main__":
    analyze_ptb_xl_classes()
