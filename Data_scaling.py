from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from data_featurization import X_train_unscaled, X_test_unscaled, X_val_unscaled

# in StandardScaler object firstly using .fit_transform() method to fit the scaler to the input data
# For subsequent uses, we have alreadly computed the statistics, we only call the .transform() method to scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.fit_transform(X_test_unscaled)
X_val = scaler.fit_transform(X_val_unscaled)


# Normalizing the scaled data
X_train = normalize(X_train)
X_test = normalize(X_test)
X_val = normalize(X_val)
