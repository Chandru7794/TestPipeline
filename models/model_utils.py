#scale the numbers her

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def scale(train, test):
    #This code scales the X data
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(train)
    X_test_scaled=scaler.fit_transform(test)
    return X_train_scaled, X_test_scaled


def train_model(X_train,y_train, X_test, y_test, alpha,model_path):
    model=Lasso(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)
    mse= mean_squared_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)

    joblib.dump(model,model_path)

    return model, {"mse":mse, "r2":r2}






