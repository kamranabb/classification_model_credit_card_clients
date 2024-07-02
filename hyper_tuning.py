from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

def hyper_tuning(clf, param_grid, x_train, y_train, x_test, y_test):
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='f1')

    # Perform Grid Search
    grid_search.fit(x_train, y_train)

    # Print the best parameters found
    best_params = grid_search.best_params_
    print("Best Parameters:", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1 Score:", f1)
    return best_model, best_params