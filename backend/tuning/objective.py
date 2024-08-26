from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np

def inner_cv_objective(trial, X_train, y_train, inner_cv,
              n_estimators_range, criterion_options, max_depth_range,
              min_samples_split_range, min_samples_leaf_range,
              max_features_range, max_samples):

    n_estimators = trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1])
    criterion = trial.suggest_categorical('criterion', criterion_options)
    max_depth = trial.suggest_int('max_depth', max_depth_range[0], max_depth_range[1])
    min_samples_split = trial.suggest_int('min_samples_split', min_samples_split_range[0], min_samples_split_range[1])
    min_samples_leaf = trial.suggest_int('min_samples_leaf', min_samples_leaf_range[0], min_samples_leaf_range[1])
    max_features = trial.suggest_int('max_features', max_features_range[0], max_features_range[1])
    max_samples = trial.suggest_float('max_samples', max_samples[0], max_samples[1])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        random_state=42
    )

    # Cross-validation logic
    scores = []
    for train_idx, val_idx in inner_cv.split(X_train, y_train):
        X_train_inner, X_val_inner = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_inner, y_val_inner = y_train.iloc[train_idx], y_train.iloc[val_idx]

        clf.fit(X_train_inner, y_train_inner.values.ravel())
        y_pred_inner = clf.predict(X_val_inner)
        scores.append(f1_score(y_val_inner, y_pred_inner, average='binary'))

    return np.mean(scores)


def outer_cv_objective(trial, folds, features,
                       n_estimators_range, criterion_options, max_depth_range,
                       min_samples_split_range, min_samples_leaf_range,
                       max_features_range, max_samples):

    n_estimators = trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1])
    criterion = trial.suggest_categorical('criterion', criterion_options)
    max_depth = trial.suggest_int('max_depth', max_depth_range[0], max_depth_range[1])
    min_samples_split = trial.suggest_int('min_samples_split', min_samples_split_range[0], min_samples_split_range[1])
    min_samples_leaf = trial.suggest_int('min_samples_leaf', min_samples_leaf_range[0], min_samples_leaf_range[1])
    max_features = trial.suggest_int('max_features', max_features_range[0], max_features_range[1])
    max_samples = trial.suggest_float('max_samples', max_samples[0], max_samples[1])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        random_state=42
    )

    # Evaluate across all outer folds
    scores = []
    for fold in folds:
        X_train, y_train = fold['X_train'][features], fold['y_train']  # Filter the features
        X_test, y_test = fold['X_test'][features], fold['y_test']  # Filter the features

        # Fit the model on the training data
        clf.fit(X_train, y_train.values.ravel())

        # Predict on the test data
        y_pred = clf.predict(X_test)

        # Calculate and store the F1 score
        scores.append(f1_score(y_test, y_pred, average='binary'))

    # Return the average F1 score across all outer folds
    return np.mean(scores)
