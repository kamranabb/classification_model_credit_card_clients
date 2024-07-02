import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr(df,label_col):

    print(df.corr()[[label_col]])

    # Visualize correlations with head map for each column
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True, ax=ax)

    # For label column
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr()[[label_col]], cmap='coolwarm', annot=True, ax=ax)

def select_feature (df, label_col, threshold = 0.1, verbose=True):
   
    # Calculate the absolute correlations
    abs_correlations = df.corr()[label_col].abs()
    abs_correlations

    # Select features with correlations exceeding the threshold
    selected_features = abs_correlations[abs_correlations > threshold].index
    df = df[selected_features]
    if verbose:
        print(df.shape)
        print(selected_features)
    return df,selected_features