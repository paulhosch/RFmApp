from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .training import train_model


def evaluate_model(folds, best_params):
    outer_results_best = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }
    outer_results_default = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }

    all_y_true, all_y_pred_best, all_y_pred_default = [], [], []

    for fold_idx, fold in enumerate(folds):
        X_train, y_train = fold['X_train'], fold['y_train']
        X_test, y_test = fold['X_test'], fold['y_test']

        # Train the model on the full training set of each fold with the best hyperparameters
        y_pred_best = train_model(X_train, y_train, X_test, best_params)

        # Train the model on the full training set of each fold with the default hyperparameters
        rf_default = RandomForestClassifier(random_state=42)  # Default parameters
        rf_default.fit(X_train, y_train.values.ravel())
        y_pred_default = rf_default.predict(X_test)

        # Store the true and predicted labels for the test set
        all_y_true.extend(y_test.values.ravel())
        all_y_pred_best.extend(y_pred_best)
        all_y_pred_default.extend(y_pred_default)

        # Calculate and store the evaluation metrics for the best params
        outer_results_best['accuracy'].append(accuracy_score(y_test, y_pred_best))
        outer_results_best['precision'].append(precision_score(y_test, y_pred_best, average='binary'))
        outer_results_best['recall'].append(recall_score(y_test, y_pred_best, average='binary'))
        outer_results_best['f1'].append(f1_score(y_test, y_pred_best, average='binary'))

        # Calculate and store the evaluation metrics for the default params
        outer_results_default['accuracy'].append(accuracy_score(y_test, y_pred_default))
        outer_results_default['precision'].append(precision_score(y_test, y_pred_default, average='binary'))
        outer_results_default['recall'].append(recall_score(y_test, y_pred_default, average='binary'))
        outer_results_default['f1'].append(f1_score(y_test, y_pred_default, average='binary'))

    return outer_results_best, outer_results_default, all_y_true, all_y_pred_best, all_y_pred_default
