import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def train_and_log_model(regression_type, degree, X_train, y_train, X_valid, y_valid, X_test, y_test):
    mlflow.set_experiment("Titanic Survival Prediction")
    with mlflow.start_run():
        if regression_type == 'Multiple Regression':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        else:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=degree)),
                ('regressor', LinearRegression())
            ])
            mlflow.log_param("Polynomial Degree", degree)

        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        mlflow.log_param("Model Type", regression_type)
        mlflow.log_metric("Cross-Validation R2", scores.mean())

        # Train final model
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "Titanic_Model")

        # Evaluate
        train_score = model.score(X_train, y_train)
        valid_score = model.score(X_valid, y_valid)
        test_score = model.score(X_test, y_test)

        mlflow.log_metric("Train R2", train_score)
        mlflow.log_metric("Valid R2", valid_score)
        mlflow.log_metric("Test R2", test_score)

        return model, train_score, valid_score, test_score

def predict_survival(model, input_data):
    return model.predict(input_data)[0]
