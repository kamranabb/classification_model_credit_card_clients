import matplotlib.pyplot as plt
import seaborn as sns

def num_stat(df):
    print(df.describe().to_string())
    print("______")
    print(df.describe)
    print("______")
    print(df.describe)
    num_cols = list(df.describe().columns)
    print(num_cols)
    print(f"Number of Numerical Columns: {len(num_cols)}")
    print(f"Number of Total Columns: {num_cols}")
    return num_cols
    
def cat_stat(df):
    # Data Description of Categorical columns
    print(df.describe(include='O'))
    print("-------------------")
    # ------------------------------------------------------
    # Print value counts of categorical columns
    # ------------------------------------------------------
    cat_cols = list(df.describe(include='O').columns)
    for column in cat_cols:
        print(df[column].value_counts().to_string())
        print("------------------")
    # Visualize value counts of categorical columns
    for column in cat_cols:
        # sns.countplot(y=column, data=df)
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column,data=df)
        plt.show()
    return cat_cols
      
    
      