"""
model.py

This script trains and evaluates multiple models (Random Forest, BERT + Logistic Regression, BERT + MLP)
for movie genre classification using plot summaries. It also allows user input for prediction using all three models.

Models Used:
- Random Forest (traditional ML)
- BERT + Logistic Regression (shallow neural model using embeddings)
- BERT + MLP (deep learning model)

Outputs:
- Evaluation metrics (accuracy, F1-score, classification report)
- Confusion matrix plots
- Misclassified instances
- Genre predictions for user-provided plot

Author : Alen K Aji
All outputs are saved in the 'models/' directory.
"""

import os
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Custom modules
from preprocess import clean_text_advanced as clean_text
from vectorizer import get_vectorizer
from random_forest import train_random_forest, predict_with_random_forest, load_random_forest_model
from bert_mlp_model import BERTMLPClassifier
from bert_lr_model import BERTClassifier

# Constants
TRAIN_PATH = os.path.join("Genre Classification Dataset", "train_data.txt")
TEST_PATH = os.path.join("Genre Classification Dataset", "test_data.txt")
TEST_LABELS_PATH = os.path.join("Genre Classification Dataset", "test_data_solution.txt")
OUTPUT_DIR = "models"
TOP_N_GENRES = 20

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Load and preprocess training data
# =====================
print("Loading training data...")
train_raw_lines = open(TRAIN_PATH, encoding='utf-8').read().splitlines()
train_data = [line.split(":::") for line in train_raw_lines if len(line.split(":::")) >= 4]
train_genres = [seg[2].strip() for seg in train_data]
train_plots = [clean_text(seg[3]) for seg in train_data]

# Identify top N genres by frequency
genre_counts = Counter(train_genres)
top_genres = [g for g, _ in genre_counts.most_common(TOP_N_GENRES)]

# Filter training data to include only top genres
train_filtered = [(g, p) for g, p in zip(train_genres, train_plots) if g in top_genres]
genres, plots = zip(*train_filtered)

# =====================
# Load and preprocess test data
# =====================
print("Loading test data...")
test_raw_lines = open(TEST_PATH, encoding='utf-8').read().splitlines()
test_label_lines = open(TEST_LABELS_PATH, encoding='utf-8').read().splitlines()

test_data = [line.split(":::") for line in test_label_lines if len(line.split(":::")) >= 4]
test_labels_clean = [seg[2].strip() for seg in test_data if seg[2].strip() in top_genres]
test_plots_clean = [clean_text(seg[3]) for seg in test_data if seg[2].strip() in top_genres]

# Safety check
if not test_labels_clean or not plots:
    raise ValueError("No data found for selected top genres. Adjust TOP_N_GENRES or check data format.")

# Assign train and test
X_train_raw, y_train_raw = plots, genres
X_test_raw, y_test_raw = test_plots_clean, test_labels_clean

# =====================
# Encode labels
# =====================
print("Label encoding...")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

# =====================
# Text vectorization (for Random Forest)
# =====================
vectorizer = get_vectorizer(max_features=10000)
print("Vectorizing text...")
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_test_vec = vectorizer.transform(X_test_raw)

# =====================
# One-hot encode labels (for neural networks)
# =====================
num_classes = len(label_encoder.classes_)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# =====================
# Compute class weights
# =====================
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# =====================
# Evaluation Function
# =====================
def evaluate_model(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)

    # Save report to file
    with open(os.path.join(OUTPUT_DIR, f"{name}_report.txt"), "w", encoding='utf-8') as f:
        f.write(f"Model: {name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\nClassification Report:\n")
        f.write(report)

    # Save misclassified instances
    misclassified = [(yt, yp, X_test_raw[i]) for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    df_mis = pd.DataFrame(misclassified, columns=['True_Label', 'Predicted_Label', 'Plot'])
    df_mis['True_Label'] = label_encoder.inverse_transform(df_mis['True_Label'])
    df_mis['Predicted_Label'] = label_encoder.inverse_transform(df_mis['Predicted_Label'])
    df_mis.to_csv(os.path.join(OUTPUT_DIR, f"{name}_misclassified.csv"), index=False)

    # Plot and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

    # Console output
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report saved.")

# =====================
# BERT + Logistic Regression
# =====================
bert_lr_model = BERTClassifier()
if os.path.exists(os.path.join(OUTPUT_DIR, "bert_logreg_model.pkl")):
    bert_lr_model.load_model()
else:
    print("\nTraining BERT + Logistic Regression...")
    bert_lr_model.fit(X_train_raw, y_train)
    y_pred_bert, y_prob_bert = bert_lr_model.predict(X_test_raw)
    evaluate_model("BERT_lr", y_test, y_pred_bert, y_prob_bert)

# =====================
# Random Forest Model
# =====================
if os.path.exists(os.path.join(OUTPUT_DIR, "random_forest_model.pkl")):
    rf_model = load_random_forest_model()
else:
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train_vec, y_train, class_weights='balanced', cv=3)
    y_pred_rf, y_prob_rf = predict_with_random_forest(rf_model, X_test_vec)
    evaluate_model("Random_Forest", y_test, y_pred_rf, y_prob_rf)

# =====================
# BERT + MLP Model
# =====================
bert_mlp = BERTMLPClassifier()
if os.path.exists(os.path.join(OUTPUT_DIR, "bert_mlp_model.h5")):
    bert_mlp.load_model()
else:
    print("\nTraining BERT + MLP...")
    bert_mlp.fit(X_train_raw, y_train, epochs=30)
    y_pred_bertmlp, y_prob_bertmlp, y_labels_bertmlp = bert_mlp.predict(X_test_raw)
    y_test_encoded = bert_mlp.label_encoder.transform(y_test)
    evaluate_model("BERT_MLP", y_test_encoded, y_pred_bertmlp, y_prob_bertmlp)

# =====================
# User Input Prediction
# =====================
print("\nEnter a movie plot to predict its genre:")
user_plot = input("Plot: ")
user_plot_clean = clean_text(user_plot)

print("\nPredictions:")

# BERT + Logistic Regression Prediction
print("- BERT + Logistic Regression:")
pred_bert, _ = bert_lr_model.predict([user_plot_clean])
print("  =>", label_encoder.inverse_transform(pred_bert)[0])

# Random Forest Prediction
print("- Random Forest:")
user_vec = vectorizer.transform([user_plot_clean])
pred_rf = rf_model.predict(user_vec)
print("  =>", label_encoder.inverse_transform(pred_rf)[0])

# BERT + MLP Prediction
print("- BERT + MLP:")
pred_bertmlp, _, y_labels_bertmlp = bert_mlp.predict([user_plot_clean])
print("  =>", y_labels_bertmlp[0])

print("\nAll model evaluations and outputs saved in 'models/' directory.")
