import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

DATA_CSV_PATH = 'data/iris.csv'
LOCAL_MODEL_OUTPUT_PATH = 'artifacts/model.joblib'
# LOCAL_METRICS_OUTPUT_PATH = 'artifacts/metrics.txt'

def train_model():
    """
    Loads iris data, trains a Decision Tree, saves model and metrics locally.
    """
    data = pd.read_csv(DATA_CSV_PATH)
    print(f"Data loaded successfully. Shape: {data.shape}")

    os.makedirs(os.path.dirname(LOCAL_MODEL_OUTPUT_PATH), exist_ok=True)

    
    X_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    y_target = 'species'

    train_df, test_df = train_test_split(data, test_size=0.4, stratify=data[y_target], random_state=42)

    X_train = train_df[X_features]
    y_train = train_df[y_target]
    X_test = test_df[X_features]
    y_test = test_df[y_target]

    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)
    print(f"Model trained: {model}")

    prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")

    joblib.dump(model,'artifacts/model.joblib')
    print(f"Trained model saved to: artifacts/model.joblib")

    # with open(LOCAL_METRICS_OUTPUT_PATH, 'w') as f:
    #     f.write(f"Accuracy: {accuracy:.4f}\n")
    #     f.write(f"Dataset rows (total loaded): {len(data)}\n") 
    # print(f"Metrics saved to: {LOCAL_METRICS_OUTPUT_PATH}")
    # return True

if __name__ == "__main__":
    print("--- Running Iris Model Training ---")
    train_model()
    print("--- Model Training and Local Saving Complete ---")
    