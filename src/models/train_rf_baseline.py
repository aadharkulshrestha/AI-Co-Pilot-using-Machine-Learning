import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
import joblib

def train_baseline(input_path: str, model_output_path: str):
    print(f"Loading windowed features from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run build_trajectory_features.py first.")
        return
        
    df = pd.read_parquet(input_path)
    
    # Target variable: abnormal_event
    # We establish the baseline using the emergency flag, but we must drop squawk
    # features from the training set to prevent trivial data leakage.
    df['abnormal_event'] = (df['squawk_7700'] > 0).astype(int)
    
    features = [
        'mean_altitude', 'min_altitude', 'max_altitude', 'altitude_std', 'altitude_change',
        'mean_groundspeed', 'min_groundspeed', 'max_groundspeed', 'speed_std', 'speed_change',
        'mean_vertical_rate', 'vertical_rate_std', 'max_climb_rate', 'max_descent_rate',
        'mean_track', 'track_std', 'track_change', 'turn_rate',
        'latitude_change', 'longitude_change'
    ]
    
    print(f"Total dataset size: {len(df)} windows")
    
    # We must split by flight_id so we don't leak adjacent windows from the same 
    # flight into both train and test sets.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['flight_id']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    X_train, y_train = train_df[features], train_df['abnormal_event']
    X_test, y_test = test_df[features], test_df['abnormal_event']
    
    print(f"Train flights: {train_df['flight_id'].nunique()}, Test flights: {test_df['flight_id'].nunique()}")
    print(f"Train windows: {len(X_train)} (Abnormal: {y_train.sum()})")
    print(f"Test windows:  {len(X_test)} (Abnormal: {y_test.sum()})")
    
    print("\nTraining Random Forest baseline model...")
    # Add class_weight='balanced' to handle potential class imbalance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("\nEvaluating model on test set...")
    y_pred = rf.predict(X_test)
    
    # Handle the case where the test set might only have 1 class
    if len(np.unique(y_test)) > 1:
        y_prob = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = float('nan')
        
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    
    if sum(y_test) > 0:
        print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
        if not np.isnan(auc):
            print(f"ROC-AUC:   {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Feature Importance
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importances.head(10).to_string(index=False))
    
    print(f"\nSaving model to {model_output_path}...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(rf, model_output_path)
    print("Baseline ML pipeline complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/trajectory_features.parquet")
    parser.add_argument("--model-output", type=str, default="models/rf_baseline.pkl")
    args = parser.parse_args()
    train_baseline(args.input, args.model_output)
