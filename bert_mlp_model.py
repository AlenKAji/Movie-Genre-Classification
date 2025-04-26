# bert_mlp_model.py
# ----------------------------------------------------------------
# BERTMLPClassifier: Extracts embeddings from BERT and trains a 
# Keras-based MLP (Multi-Layer Perceptron) for text classification.
# Features:
# - GPU support via PyTorch BERT backend
# - Embedding batching with progress bar
# - Keras model training with categorical labels
# - Save/load model and label encoder with relative paths
# ----------------------------------------------------------------

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

class BERTMLPClassifier:
    def __init__(self, model_name='bert-base-uncased', freeze_bert=True):
        """
        Initialize BERT tokenizer and model, set up device and MLP model container.
        Optionally freeze BERT weights to prevent fine-tuning.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name).to(self.device)
        self.bert.eval()  # Use BERT in inference mode

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False  # Freeze BERT parameters

        self.model = None
        self.label_encoder = None

        # Paths to save/load model and label encoder
        self.model_path = os.path.join("models", "bert_mlp_model.h5")
        self.encoder_path = os.path.join("models", "bert_mlp_encoder.pkl")

    def get_embeddings(self, texts, batch_size=32):
        """
        Convert input texts into BERT CLS embeddings using batching.

        Args:
            texts (list): Input text samples.
            batch_size (int): Number of samples per batch.

        Returns:
            np.ndarray: BERT CLS embeddings.
        """
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with BERT"):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.bert(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract [CLS] token
                embeddings.extend(cls_embeddings)
        return np.array(embeddings)

    def build_mlp(self, input_dim, output_dim):
        """
        Build and compile the MLP model architecture.

        Args:
            input_dim (int): Size of BERT embedding.
            output_dim (int): Number of output classes.

        Returns:
            keras.Model: Compiled MLP model.
        """
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X_texts, y, epochs=15, callbacks=None):
        """
        Train the BERT + MLP model on input texts and labels.

        Args:
            X_texts (list): Input text samples.
            y (list): Class labels.
            epochs (int): Training epochs.
            callbacks (list): Optional Keras callbacks.
        """
        print("📦 Extracting BERT embeddings...")
        X_embed = self.get_embeddings(X_texts)

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)

        self.model = self.build_mlp(X_embed.shape[1], y_categorical.shape[1])

        print("🧠 Training MLP on BERT embeddings...")
        self.model.fit(
            X_embed, y_categorical,
            epochs=epochs,
            batch_size=64,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )

        os.makedirs("models", exist_ok=True)
        self.model.save(self.model_path)
        joblib.dump(self.label_encoder, self.encoder_path)
        print(f"✅ Model saved to {self.model_path}")

    def predict(self, X_texts):
        """
        Predict class labels from input text samples.

        Args:
            X_texts (list): Input texts.

        Returns:
            tuple: Predicted label indices, class probabilities, and label names.
        """
        X_embed = self.get_embeddings(X_texts)
        y_prob = self.model.predict(X_embed)
        y_pred = np.argmax(y_prob, axis=1)
        y_labels = self.label_encoder.inverse_transform(y_pred)
        return y_pred, y_prob, y_labels

    def evaluate(self, X_texts, y_true):
        """
        Evaluate model accuracy and F1 score.

        Args:
            X_texts (list): Input texts.
            y_true (list): Ground truth labels.

        Returns:
            tuple: Accuracy and weighted F1-score.
        """
        y_pred, _, _ = self.predict(X_texts)
        y_true_enc = self.label_encoder.transform(y_true)
        acc = accuracy_score(y_true_enc, y_pred)
        f1 = f1_score(y_true_enc, y_pred, average='weighted')
        return acc, f1

    def load_model(self):
        """
        Load the trained MLP model and label encoder from disk.
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert = BertModel.from_pretrained(self.model_name).to(self.device)
        self.bert.eval()

        self.model = load_model(self.model_path)
        self.label_encoder = joblib.load(self.encoder_path)

        print("🧠 DEBUG: class labels loaded ➝", self.label_encoder.classes_)
        print(f"✅ Model loaded from {self.model_path}")
