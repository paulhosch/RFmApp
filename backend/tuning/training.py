from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, X_test, best_params):
    clf = RandomForestClassifier(**best_params, random_state=42)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_test)
    return y_pred
