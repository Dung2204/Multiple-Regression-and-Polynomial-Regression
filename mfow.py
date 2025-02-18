import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Kiểm tra thư mục lưu trữ MLflow
mlruns_path = os.path.abspath("mlruns")
if not os.path.exists(mlruns_path):
    os.makedirs(mlruns_path)

# Thiết lập MLflow tracking URI
mlflow.set_tracking_uri("file:///" + mlruns_path)

# Đọc dữ liệu Titanic
df = sns.load_dataset("titanic").dropna()
X = df[['pclass', 'age', 'sibsp', 'parch', 'fare']]
y = df['survived']

# Chia dữ liệu thành tập train/test
random_state = 42
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Giao diện Streamlit
st.title("🚢 Titanic Model Training & Analysis")

if st.button("🚀 Train Model"):
    with mlflow.start_run():
        # Ghi lại tham số mô hình
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)

        # Huấn luyện mô hình RandomForest
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)

        # Dự đoán và tính độ chính xác
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Ghi lại độ chính xác vào MLflow
        mlflow.log_metric("accuracy", acc)

        # Lưu mô hình vào MLflow
        model_path = "Titanic_Model"
        mlflow.sklearn.log_model(model, model_path)

        # Hiển thị thông tin trong Streamlit
        st.success(f"✅ Mô hình đã được huấn luyện!")
        st.write(f"🔹 **Độ chính xác**: `{acc:.4f}`")
        st.write(f"📂 **Mô hình đã được lưu tại MLflow:** `{mlruns_path}`")
        st.write(f"📌 **Tên mô hình**: `{model_path}`")

        print(f"Độ chính xác của mô hình: {acc:.4f}")
        print(f"Mô hình đã được lưu tại: {mlruns_path}/{model_path}")

st.info("Nhấn 'Train Model' để huấn luyện mô hình và lưu vào MLflow.")
