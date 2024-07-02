import pandas as pd
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


def missing_value(df):
    print("Missing Values:")
    print(df.isnull().sum())
    print("------------------")

    # ------------------------------------------------------
    # Calculate the percentage of missing values in each column of df
    # ------------------------------------------------------
    print("Percentage of Missing Values:")
    dfpervalue=df.isnull().sum()*100/len(df)
    print(dfpervalue)
    
    
def data_types(df):
    print(df.dtypes)
    
    
def remove_col_label(num_cols, cat_cols,label_col):

    if label_col in num_cols:
        num_cols=num_cols.remove(label_col)
    elif label_col in cat_cols:
        cat_cols=cat_cols.remove(label_col)
    return num_cols,cat_cols  


def fix_missing_values(df, label_col, numerical_columns):
    # ------------------------------------------------------
    # Fixing Missing Values
    # ------------------------------------------------------
    df_fixed=df.copy()
    cols_to_fix = df_fixed.isnull().sum()[df_fixed.isnull().sum() != 0].index

    if len(cols_to_fix) != 0:
        # Remove null values from label column
        df_fixed = df_fixed[df_fixed[label_col].notnull()]

        # Remove label column from cols_to_fix if there is
        if label_col in cols_to_fix:
            cols_to_fix = cols_to_fix.drop(label_col)

        # Fill missing values with mean for numerical columns
        numerical_cols_to_fix = cols_to_fix.intersection(numerical_columns)
        if numerical_cols_to_fix.size > 0:
            df_fixed[numerical_cols_to_fix] = df_fixed[numerical_cols_to_fix].fillna(df_fixed[numerical_cols_to_fix].mean())

        # Fill missing values with mode for categorical columns
        categorical_cols_to_fix = cols_to_fix.drop(numerical_cols_to_fix)
        if categorical_cols_to_fix.size > 0:
            df_fixed[categorical_cols_to_fix] = df_fixed[categorical_cols_to_fix].fillna(df_fixed[categorical_cols_to_fix].mode()[0])

        # ------------------------------------------------------
        # Check for missing values
        # ------------------------------------------------------
        print("Missing Values:")
        print(df_fixed.isnull().sum())
        print("------------------")

        # ------------------------------------------------------
        # Calculate the percentage of missing values in each column of df
        # ------------------------------------------------------
        print("Percentage of Missing Values:")
        dfpervalue=df_fixed.isnull().sum()*100/len(df)
        print(dfpervalue)
    else:
        print("There is no missing value in the dataset.")
    return df_fixed




def dummied_cols(df, columns, label_col):
    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df, columns=columns)
    df_encoded = df_encoded * 1
    
    # Reorder columns
    label_index = df_encoded.columns.get_loc(label_col)
    cols = list(df_encoded.columns)
    cols.pop(label_index)
    cols.append(label_col)
    df_encoded = df_encoded[cols]

    # Extract only the dummied columns
    dummied_columns = [col for col in df_encoded.columns if col not in df.columns]
    return df_encoded, dummied_columns 





def standard_scaler(df, X, y, label_col, dummied_columns):
    to_scale = list(set(X.columns) - set(dummied_columns))  # Corrected line
    scaler = StandardScaler()
    inputsscaled = scaler.fit_transform(df[to_scale])
    standardscaler_df = pd.DataFrame(inputsscaled, columns=to_scale)
    
    
    to_add = [col for col in X.columns if col in dummied_columns]
   
    if to_add:
        for col in to_add:
            standardscaler_df[col] = df[col]  # Add dummy columns back
        
    standardscaler_df[label_col] = y.values  # Add target column back
    print(standardscaler_df.head().to_string())
    
    return standardscaler_df


def robust_scaler(df, X, y, label_col, dummied_columns):
    to_scale = list(set(X.columns) - set(dummied_columns))  # Corrected line
    scaler = RobustScaler()
    inputsscaled = scaler.fit_transform(df[to_scale])
    robust_df = pd.DataFrame(inputsscaled, columns=to_scale)
    
    
    to_add = [col for col in X.columns if col in dummied_columns]
   
    if to_add:
        for col in to_add:
            robust_df[col] = df[col]  # Add dummy columns back
        
    robust_df[label_col] = y.values  # Add target column back
    print(robust_df.head().to_string())
    
    return robust_df

def maxabs_scaler(df, X, y, label_col, dummied_columns):
    to_scale = list(set(X.columns) - set(dummied_columns))  # Corrected line
    scaler = MaxAbsScaler()
    inputsscaled = scaler.fit_transform(df[to_scale])
    maxabsscaler_df = pd.DataFrame(inputsscaled, columns=to_scale)
    
    
    to_add = [col for col in X.columns if col in dummied_columns]
   
    if to_add:
        for col in to_add:
            maxabsscaler_df[col] = df[col]  # Add dummy columns back
        
    maxabsscaler_df[label_col] = y.values  # Add target column back
    print(maxabsscaler_df.head().to_string())
    
    return maxabsscaler_df

def quantile_transformer(df, X, y, label_col, dummied_columns):
    to_scale = list(set(X.columns) - set(dummied_columns))  # Corrected line
    scaler = QuantileTransformer()
    inputsscaled = scaler.fit_transform(df[to_scale])
    quantiletrans_df = pd.DataFrame(inputsscaled, columns=to_scale)
    
    
    to_add = [col for col in X.columns if col in dummied_columns]
   
    if to_add:
        for col in to_add:
            quantiletrans_df[col] = df[col]  # Add dummy columns back
        
    quantiletrans_df[label_col] = y.values  # Add target column back
    print(quantiletrans_df.head().to_string())
    
    return quantiletrans_df


def power_transformer(df, X, y, label_col, dummied_columns):
    to_scale = list(set(X.columns) - set(dummied_columns))  # Corrected line
    scaler = PowerTransformer()
    inputsscaled = scaler.fit_transform(df[to_scale])
    powertrans_df = pd.DataFrame(inputsscaled, columns=to_scale)
    
    
    to_add = [col for col in X.columns if col in dummied_columns]
   
    if to_add:
        for col in to_add:
            powertrans_df[col] = df[col]  # Add dummy columns back
        
    powertrans_df[label_col] = y.values  # Add target column back
    print(powertrans_df.head().to_string())
    
    return powertrans_df

def minmax_scaler(df, X, y, label_col, dummied_columns):
    to_scale = list(set(X.columns) - set(dummied_columns))  # Corrected line
    scaler = MinMaxScaler()
    inputsscaled = scaler.fit_transform(df[to_scale])
    minmaxscaler_df = pd.DataFrame(inputsscaled, columns=to_scale)
    
    
    to_add = [col for col in X.columns if col in dummied_columns]
   
    if to_add:
        for col in to_add:
            minmaxscaler_df[col] = df[col]  # Add dummy columns back
        
    minmaxscaler_df[label_col] = y.values  # Add target column back
    print(minmaxscaler_df.head().to_string())
    
    return minmaxscaler_df


def l2_normalizer(df, X, y, label_col, dummied_columns):
    to_scale = list(set(X.columns) - set(dummied_columns))  # Corrected line
    scaler = Normalizer(norm='l2')
    inputsscaled = scaler.fit_transform(df[to_scale])
    l2_df = pd.DataFrame(inputsscaled, columns=to_scale)
    
    
    to_add = [col for col in X.columns if col in dummied_columns]
   
    if to_add:
        for col in to_add:
            l2_df[col] = df[col]  # Add dummy columns back
        
    l2_df[label_col] = y.values  # Add target column back
    print(l2_df.head().to_string())
    
    return l2_df


def l1_normalizer(df, X, y, label_col, dummied_columns):
    to_scale = list(set(X.columns) - set(dummied_columns))  # Corrected line
    scaler = Normalizer(norm='l1')
    inputsscaled = scaler.fit_transform(df[to_scale])
    l1_df = pd.DataFrame(inputsscaled, columns=to_scale)
    
    
    to_add = [col for col in X.columns if col in dummied_columns]
   
    if to_add:
        for col in to_add:
            l1_df[col] = df[col]  # Add dummy columns back
        
    l1_df[label_col] = y.values  # Add target column back
    print(l1_df.head().to_string())
    
    return l1_df

