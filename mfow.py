import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Bật chế độ tự động log (có thể bỏ nếu muốn tự log từng giá trị)
mlflow.autolog()

# Đọc dữ liệu Titanic
df = sns.load_dataset("titanic").dropna()
X = df[['pclass', 'age', 'sibsp', 'parch', 'fare']]
y = df['survived']

# Chia dữ liệu thành tập train/test
random_state = 42
test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Khởi tạo MLflow
with mlflow.start_run():
    # Ghi lại tham số mô hình
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", test_size)

    # Huấn luyện mô hình RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Dự đoán trên tập test
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Ghi lại độ chính xác vào MLflow
    mlflow.log_metric("accuracy", acc)

    # Lưu mô hình vào MLflow
    mmlflow.sklearn.log_model(model, "Titanic_Model", artifact_path="./mlflow_artifacts")


    print(f"Độ chính xác của mô hình: {acc:.4f}")
    print("Mô hình đã được lưu trong MLflow.")
