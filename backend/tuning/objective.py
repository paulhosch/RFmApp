from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np

def inner_cv_objective(trial, X_train, y_train, inner_cv, param_ranges):
    params = {}
    for param, value in param_ranges.items():
        if isinstance(value, list):
            params[param] = trial.suggest_categorical(param, value)
        elif isinstance(value, tuple) and len(value) == 2:
            if isinstance(value[0], int):
                params[param] = trial.suggest_int(param, value[0], value[1])
            elif isinstance(value[0], float):
                params[param] = trial.suggest_float(param, value[0], value[1])
        else:
            params[param] = value

    clf = RandomForestClassifier(**params, random_state=42)

    # Cross-validation logic
    scores = []
    for train_idx, val_idx in inner_cv.split(X_train, y_train):
        X_train_inner, X_val_inner = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_inner, y_val_inner = y_train.iloc[train_idx], y_train.iloc[val_idx]

        clf.fit(X_train_inner, y_train_inner.values.ravel())
        y_pred_inner = clf.predict(X_val_inner)
        scores.append(f1_score(y_val_inner, y_pred_inner, average='binary'))

    return np.mean(scores)

def outer_cv_objective(trial, folds, features, param_ranges):
    params = {}
    for param, value in param_ranges.items():
        if isinstance(value, list):
            params[param] = trial.suggest_categorical(param, value)
        elif isinstance(value, tuple) and len(value) == 2:
            if isinstance(value[0], int):
                params[param] = trial.suggest_int(param, value[0], value[1])
            elif isinstance(value[0], float):
                params[param] = trial.suggest_float(param, value[0], value[1])
        else:
            params[param] = value

    clf = RandomForestClassifier(**params, random_state=42)

    # Evaluate across all outer folds
    scores = []
    for fold in folds:
        X_train, y_train = fold['X_train'][features], fold['y_train']
        X_test, y_test = fold['X_test'][features], fold['y_test']

        clf.fit(X_train, y_train.values.ravel())
        y_pred = clf.predict(X_test)
        scores.append(f1_score(y_test, y_pred, average='binary'))

    return np.mean(scores)