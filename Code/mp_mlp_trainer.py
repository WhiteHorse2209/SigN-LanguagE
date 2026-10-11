import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'landmarks_data_2hands.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'mp_mlp_model.pkl')

def train_model():
    print("Loading landmark data...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: '{CSV_PATH}' not found. Run mp_dataset_creator.py or mp_record_live.py first!")
        return

    data = pd.read_csv(CSV_PATH)
    
    X = data.drop('label', axis=1).values
    y = data['label'].values
    
    print(f"Dataset: {len(data)} frames, {len(np.unique(y))} classes: {np.unique(y)}")

    # Encode labels to numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Normalize features - very important for MLP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    print("Training MLP Neural Network...")
    print("Architecture: Input -> 256 -> 128 -> 64 -> Output")
    
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,           # Stop if no improvement
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=True
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("[Done] MLP Training Complete!")
    print(f"Model Accuracy: {acc*100:.2f}%")
    print("="*40)
    print("\nClassification Report:")
    # Decode labels back to names for readability
    y_test_names = le.inverse_transform(y_test)
    y_pred_names = le.inverse_transform(y_pred)
    print(classification_report(y_test_names, y_pred_names))

    # Save model, scaler and label encoder together
    bundle = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    
    print(f"[Success] MLP Model saved as '{MODEL_PATH}'")

if __name__ == "__main__":
    train_model()
