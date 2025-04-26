"""
random_forest_model.py

This module handles:
- Training a Random Forest model with a pipeline
- Saving and loading the model
- Making predictions

"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

def train_random_forest(X_train, y_train, class_weights='balanced', cv=5):
    """
    Trains a Random Forest classifier using a pipeline that includes:
    - Feature scaling
    - Feature selection
    - Classifier with hyperparameter tuning (GridSearchCV)

    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training labels.
        class_weights (str or dict, optional): Class weight balancing. Default is 'balanced'.
        cv (int, optional): Number of cross-validation folds. Default is 5.

    Returns:
        sklearn model: Best trained Random Forest model.
    """
    # Ensure model directory exists
    os.makedirs("models", exist_ok=True)

    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight=class_weights,
                random_state=42
            )
        )),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define hyperparameters for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__max_features': ['sqrt'],
        'classifier__class_weight': [class_weights]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
    )

    # Train the model
    grid_search.fit(X_train, y_train)

    # Save the best model
    model_path = os.path.join("models", "random_forest_model.pkl")
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"✅ Random Forest model saved to {model_path}")

    return grid_search.best_estimator_

def predict_with_random_forest(model, X_test):
    """
    Makes predictions and probability estimates using a trained Random Forest model.

    Args:
        model (sklearn model): Trained Random Forest model.
        X_test (array-like): Test feature data.

    Returns:
        tuple: (predictions, probabilities)
    """
    return model.predict(X_test), model.predict_proba(X_test)

def load_random_forest_model(model_path="models/random_forest_model.pkl"):
    """
    Loads a saved Random Forest model from the specified path.

    Args:
        model_path (str, optional): Path to the saved model file. Default is 'models/random_forest_model.pkl'.

    Returns:
        sklearn model: Loaded Random Forest model.
    """
    model = joblib.load(model_path)
    print(f"✅ Random Forest model loaded from {model_path}")
    return model

# END OF random_forest_model.py
