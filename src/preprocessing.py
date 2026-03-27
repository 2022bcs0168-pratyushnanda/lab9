import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(X_train, X_test, scale=True):
    # Handle missing values (total_bedrooms has NaNs)
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # One-hot encode 'ocean_proximity'
    if 'ocean_proximity' in X_train.columns:
        X_train = pd.get_dummies(X_train, columns=['ocean_proximity'], drop_first=True)
        X_test = pd.get_dummies(X_test, columns=['ocean_proximity'], drop_first=True)
        # Align columns (test may be missing some categories)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    if not scale:
        return X_train, X_test

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

