import matplotlib.pyplot as plt
import seaborn as sns


def scatter(df):
    label_col = 'default'
    x_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]
    y = df[label_col]

    for col in x_cols:
        x = df[col]
        plt.scatter(x,y, color='green')
        plt.xlabel(col)
        plt.ylabel(label_col)
        plt.show()
        
        
def bar(df,cat_cols):
    x_cols = cat_cols + ['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    y_col = 'LIMIT_BAL'
    # top_n = 15

    for x_col in x_cols:
        groupby_df = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        feature = groupby_df.index
        label = groupby_df.values

        plt.figure(figsize=(10, 6))
        plt.bar(feature, label, color='green')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{x_col} vs {y_col}')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

        # Add numbers over the bar plots
        # for i, v in enumerate(label):
        #     plt.text(i, v, "{:.2f}".format(v), color='black', fontweight='bold', ha='center')

        plt.show()
        
def boxplot(df,num_cols):
    def plot_boxplot(cols, df):
        for col in cols:
            sns.boxplot(x=df[col], data=df, color='green')
            plt.show()

    n_cols = 5
    boxplot_cols = num_cols[:n_cols]
    plot_boxplot(boxplot_cols, df)
    
def plt_histogram(df,col):
    
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], color = 'green')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
    
def sns_histogram(df,col1,col2):
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col1, hue=col2, multiple='stack', kde=True, color='green')
    plt.xlabel(col1)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col1} by {col2}')
    plt.show()
    
def density_plot(df,col):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df[col], shade=True, color='green')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.title(f'Density Plot of {col}')
    plt.show()
    
def distribution(df,vis_col):
    sns.distplot(df[vis_col], color = 'green', hist_kws={'alpha': 0.4}, bins=10)
# sns.distplot(df[col], color = 'green', hist_kws={'alpha': 0.4}, bins=10, kde=False)

def joint(df,joinplot_cols,vis_col):
    for col in joinplot_cols:
        sns.set(style="whitegrid")
        g = sns.jointplot(x=col, y=vis_col, data=df, kind="scatter", color = 'green')
        # g = sns.jointplot(x=col, y=vis_col, data=df, kind="reg", color = 'green')
        g.set_axis_labels(col, vis_col)
        g.fig.suptitle(f"Joint Plot of {col} vs. {vis_col}")
        plt.show()
        
def pairplots(df, columns_to_include,vis_col):
    
    sns.set(style="ticks")
    # sns.pairplot(df[columns_to_include])
    # sns.pairplot(df[columns_to_include], diag_kind="kde", palette='colorblind')
    sns.pairplot(df[columns_to_include], diag_kind="kde", hue='MARRIAGE', palette='colorblind')
    plt.show()
    
    