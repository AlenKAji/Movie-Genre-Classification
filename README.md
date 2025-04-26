# 🎬 Movie Genre Classification

## ✨ Task Objectives

The objective of this project is to build a **multi-model movie genre classification system** based on **plot summaries**.  
The system leverages both traditional machine learning techniques and modern deep learning approaches to classify plots into genres.

Models implemented:
- **Random Forest** with TF-IDF vectorization and handcrafted features
- **BERT + Logistic Regression** (shallow neural model with BERT embeddings)
- **BERT + MLP** (deep neural network on BERT embeddings using Keras)

The project emphasizes:
- Comparative analysis between classical ML and BERT-based models
- Handling class imbalance with appropriate strategies
- Saving and visualizing model performances
- User interaction to predict genre from custom plot input

---

## 📁 Project Structure

```
├── model.py                  # Main script to train, evaluate, and predict with all models
├── bert_lr_model.py         # BERT + Logistic Regression model
├── bert_mlp_model.py        # BERT + MLP model (Keras)
├── random_forest.py         # Random Forest pipeline with GridSearchCV
├── vectorizer.py            # TF-IDF + handcrafted features
├── preprocess.py            # Text preprocessing and cleaning
├── requirements.txt         # Required Python packages
├── models/                  # Directory for saving trained models and evaluation outputs
├── Genre Classification Dataset/
│   ├── train_data.txt
│   ├── test_data.txt
│   └── test_data_solution.txt
```

---
## 🛠 Steps to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/movie-genre-classification.git
   cd movie-genre-classification
   ```

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**

   Make sure the following files exist in a folder called `Genre Classification Dataset/`:
   - `train_data.txt`
   - `test_data.txt`
   - `test_data_solution.txt`

4. **Run the project**
   ```bash
   python model.py
   ```

5. **View outputs**

   All results (saved models, confusion matrices, misclassified examples, classification reports) will be automatically saved in the `models/` directory.

6. **Make custom predictions**

   After evaluation, the system will prompt you to input your own movie plot, and each model will predict the genre.

---

## 🛠 Requirements

- Python 3.10
- See `requirements.txt` for full list

---

## 🧹 Code Quality

- **Clean architecture**: Code is modularized across multiple files (`bert_lr_model.py`, `random_forest.py`, etc.)
- **Well-commented**: Each function is documented with docstrings and inline comments
- **Reproducible**: Fixed model saving paths ensure reproducibility
- **Scalable**: Easy to plug new models or vectorizers into the pipeline

---

## ✍️ Author

**Alen K Aji**

---

## 📄 License

This project is open-source under the MIT License </br>
Copyright (c) 2025 Alen K Aji
