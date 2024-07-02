# ---------------------------------------
# Libraries
# ---------------------------------------
from utils.load_data import *
from utils.statistics import *
from utils.preprocessing import *
from utils.visualization import *
from utils.feature_engineering import *
from utils.train import*
from utils.hyper_tuning import *
from utils.evaluate import *
from utils.save_load_model import *


# ---------------------------------------
# Constants
# ---------------------------------------
label_col='default'
vis_col = 'LIMIT_BAL'
hue_col = 'SEX'
xls_path = 'data/default of credit card clients.xls'

# ---------------------------------------
# Load and Explore data
# ---------------------------------------
df = load_xls(xls_path)


# ---------------------------------------
# Explore data
# ---------------------------------------
num_cols = num_stat(df)
cat_cols = cat_stat(df)
missing_value(df)
data_types(df)

# ---------------------------------------
# EDA
# ---------------------------------------
# scatter(df)
# bar(df,cat_cols)
# boxplot(df,num_cols)
# plt_histogram(df,vis_col)
# sns_histogram(df,vis_col,hue_col)
# density_plot(df,vis_col)
# distribution(df,vis_col)

# joinplot_cols = ["AGE", "BILL_AMT1", "PAY_AMT1"]
# joint(df,joinplot_cols,vis_col)

# columns_to_include = ["AGE", "BILL_AMT1", "PAY_AMT1", "MARRIAGE"] + [vis_col]
# pairplots(df, columns_to_include,vis_col)

# ---------------------------------------
# Preprocessing
# ---------------------------------------
remove_col_label(num_cols, cat_cols,label_col)    
num_cols, cat_cols = remove_col_label(num_cols, cat_cols,label_col) 
df_fixed = fix_missing_values(df, label_col, num_cols)
df_dummied, dummied_columns= dummied_cols(df_fixed, cat_cols,label_col)

# ---------------------------------------
# Feature Engineering
# ---------------------------------------
columns_corr= plot_corr(df_dummied,label_col)
df_final, selected_features = select_feature (df_dummied, label_col, threshold = 0.1, verbose=True)
X = df_final[selected_features]
X = X.drop(label_col, axis=1)
y = df_final[label_col]

# ---------------------------------------
# Scaling & Normalization
# ---------------------------------------
standardscaler_df = standard_scaler(df_final,X,y,label_col,dummied_columns)
robustscaler_df = robust_scaler(df, X, y, label_col, dummied_columns)
maxabsscaler_df = maxabs_scaler(df, X, y, label_col, dummied_columns)
quantile_transformer_df = quantile_transformer(df, X, y, label_col, dummied_columns)
power_transformer_df = power_transformer(df, X, y, label_col, dummied_columns)
minmax_scaler_df = minmax_scaler(df, X, y, label_col, dummied_columns)
l2_normalizer_df = l2_normalizer(df, X, y, label_col, dummied_columns)
l1_normalizer_df = l1_normalizer(df, X, y, label_col, dummied_columns)

# ---------------------------------------
# Split data
# ---------------------------------------
x_train,x_test,y_train,y_test = split(X,y)

# ---------------------------------------
# SMOTE
# ---------------------------------------
x_train_resampled,y_train_resampled = smoting(x_train, y_train)

# ---------------------------------------
# Algorithm Selection
# ---------------------------------------
names = ['MLPClassifier','Linear SVC', 'Stochastic Gradient Descent (SGD)', 'Gradient Boosting', 'Logistic Regression', 
    'Decision Tree','K Nearest Neighbors', 'Support Vector Classification (SVC)', 'Random Forest','Extremely Randomized Trees',
    'Averaged Perceptron Classifier',  'Gaussian Naive Bayes', 'GaussianProcessClassifier','AdaBoostClassifier', 'XGBClassifier']

classifiers = [ MLPClassifier(), LinearSVC(), SGDClassifier(), GradientBoostingClassifier(), LogisticRegression(), 
    DecisionTreeClassifier(), KNeighborsClassifier(), SVC(), RandomForestClassifier(), ExtraTreesClassifier(), Perceptron(), GaussianNB(), GaussianProcessClassifier(), AdaBoostClassifier(), XGBClassifier()]

algo_selection(x_train_resampled, y_train_resampled, x_test, y_test, names, classifiers)


# ---------------------------------------
# Hyperparameter Tuning
# ---------------------------------------
param_grid = {
    'n_estimators': [50],
    'learning_rate': [0.1],
    'max_depth': [7]
}

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7]
# }


clf = GradientBoostingClassifier()
best_model, best_params = hyper_tuning(clf, param_grid, x_train_resampled, y_train_resampled, x_test, y_test)

# ---------------------------------------
# Evaluation
# ---------------------------------------
evaluate(best_model, x_test, y_test)

# ---------------------------------------
# Save Model
# ---------------------------------------
model_path = 'models/best_model.pkl'
save_model(best_model, model_path)

# ---------------------------------------
# Load Model
# ---------------------------------------
loaded_model = load_model(model_path)

# ---------------------------------------
# Inference
# ---------------------------------------
n_pred = 50
x_input=x_test.head(n_pred)
y_pred=loaded_model.predict(x_input)

for idx, pred in enumerate(y_pred):
    print("Prediction:", y_pred[idx])
    print("Ground Truth:", y_test.iloc[idx])
    if y_pred[idx]==y_test.iloc[idx]:
        print("Predictions is correct")
    else:
        print("Wrong prediction")
    print('-'*20)
    
