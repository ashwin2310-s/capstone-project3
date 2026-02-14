import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix)

#1.GLOBAL SETTINGS
RANDOM_SEED = 42
TEST_SIZE = 0.2
FILE_PATH = "C:/Users/ashwi/Downloads/sql session 1/synthetic_food_dataset_imbalanced.csv"

# 2. DATA LOADING & INITIAL EXPLORATION
def load_and_explore(path):
    print("--- Phase 1: Loading & Exploring Data ---")
    df = pd.read_csv(path)
    
    df.columns = df.columns.str.strip()
    
    print(f"Dataset Loaded Successfully. Shape: {df.shape}")
    print("\nColumn Names found:", list(df.columns))
    
    if 'Food_Name' not in df.columns:
        print("Warning: 'Food_Name' column not found. Please check column names.")
        return None
        
    print("\nClass Distribution (Top 10 items):\n", df['Food_Name'].value_counts().head(10))
    return df

# 3. DATA PREPROCESSING & CLEANING
def preprocess_data(df):
    print("\n--- Phase 2: Data Preprocessing ---")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    nutrients = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar'] 
    for col in nutrients:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"Cleaned Data Shape: {df.shape}")
        
    le = LabelEncoder()
    df['Food_Encoded'] = le.fit_transform(df['Food_Name']) 
    
    return df, le

# 4. MODEL TRAINING & EVALUATION
def train_model(df):
    print("\n--- Phase 3: Model Training & Validation ---")
    
    # X features match your CSV: 'Carbs'
    X = df[['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar']]
    y = df['Food_Encoded']

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    
    # Cross Validation
    print("Performing 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    
    # Final Fit
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    print("\n--- Evaluation Results ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nReport:\n", classification_report(y_test, y_pred))
    
    return rf_model, scaler, X_test_scaled, y_test, y_pred

#5.VISUALIZATIONS
def generate_visuals(model, X_test, y_test, y_pred, le, features):
    print("\n--- Phase 4: Visualizations ---")
    
    #Confusion Matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm[:15, :15], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Top 15 Food Categories)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    #Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=features, palette='magma')
    plt.title('Feature Importance: Which Nutrient Defines the Food?')
    plt.show()

# ---EXECUTION ---
if __name__ == "__main__":
    try:
        #1.Load
        food_data = load_and_explore(FILE_PATH)
        
        if food_data is not None:
            #2.Preprocess
            clean_data, label_encoder = preprocess_data(food_data)
            
            #3.Train
            trained_model, data_scaler, X_test_s, y_test_true, y_pred_val = train_model(clean_data)
            
            #4.Visuals (Passing 'Carbs' as feature label)
            generate_visuals(trained_model, X_test_s, y_test_true, y_pred_val, 
                             label_encoder, ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar'])
            
            #5.Custom Prediction Function
            def predict_meal(cals, prot, fat, carb, sugar):
                sample = data_scaler.transform([[cals, prot, fat, carb, sugar]])
                idx = trained_model.predict(sample)
                return label_encoder.inverse_transform(idx)[0]

            print("\n--- Manual Prediction Test ---")
            result = predict_meal(250, 20, 10, 15, 5)
            print(f"Input: 250 Cal, 20g Protein -> Predicted Food: {result}")

    except Exception as e:
        print(f"Error during execution: {e}")