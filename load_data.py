import pandas as pd

def load_xls(path, verbose=True):
    df = pd.read_excel(path)
    
    # Extract number of rows and columns
    num_rows, num_cols = df.shape

    if verbose:
        print(list(df.columns))
        print("------------------")
        print(df.head().to_string())
        # Print number of rows and columns in a pretty way
        print(f"Number of Rows: {num_rows}")
        print(f"Number of Columns: {num_cols}")
    return df
