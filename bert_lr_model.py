# bert_model.py
# --------------------------------------------------------
# BERTClassifier: Extracts text embeddings using BERT and 
# trains a Logistic Regression model for classification.
# Features:
# - Batching and GPU acceleration support
# - Model saving and loading (relative paths)
# --------------------------------------------------------

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

class BERTClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize BERT tokenizer, BERT model, and Logistic Regression classifier.
        Automatically uses GPU if available.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name).to(self.device)
        self.bert.eval()  # Set BERT to evaluation mode (no weight updates)

        self.classifier = LogisticRegression(
            max_iter=1000,    # Allow more iterations for convergence
            solver='saga',    # Better solver for large, sparse datasets
            verbose=1,        # Enable progress messages
            n_jobs=-1         # Use all available CPU cores
        )
        
        # Model save path (relative)
        self.model_save_path = os.path.join("models", "bert_logreg_model.pkl")

    def get_embeddings(self, texts, batch_size=16):
        """
        Generate CLS token embeddings for a list of texts using BERT.
        
        Args:
            texts (list): List of input texts.
            batch_size (int): Number of samples per batch.

        Returns:
            np.ndarray: Array of BERT embeddings.
        """
        embeddings = []
        with torch.no_grad():  # Disable gradient computation
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with BERT"):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.bert(**inputs)
                
                # Extract the [CLS] token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings)
        
        return np.array(embeddings)

    def fit(self, X_texts, y):
        """
        Train the Logistic Regression classifier on text data.

        Args:
            X_texts (list): Input texts for training.
            y (list or np.ndarray): Labels for training.
        """
        X_embed = self.get_embeddings(X_texts)
        self.classifier.fit(X_embed, y)
        
        # Save the trained classifier
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(self.classifier, self.model_save_path)
        print(f"✅ Model saved to {self.model_save_path}")

    def predict(self, X_texts):
        """
        Predict labels for new texts.

        Args:
            X_texts (list): Input texts for prediction.

        Returns:
            tuple: Predicted labels and their probabilities.
        """
        X_embed = self.get_embeddings(X_texts)
        y_pred = self.classifier.predict(X_embed)
        y_prob = self.classifier.predict_proba(X_embed)
        return y_pred, y_prob

    def evaluate(self, X_texts, y_true):
        """
        Evaluate model performance on given test data.

        Args:
            X_texts (list): Input texts.
            y_true (list or np.ndarray): True labels.

        Returns:
            tuple: Accuracy and weighted F1-score.
        """
        y_pred, _ = self.predict(X_texts)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return acc, f1

    def load_model(self):
        """
        Load the pretrained BERT model and the saved Logistic Regression classifier.
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert = BertModel.from_pretrained(self.model_name).to(self.device)
        self.bert.eval()
        
        self.classifier = joblib.load(self.model_save_path)
        print(f"✅ Model loaded from {self.model_save_path}")
