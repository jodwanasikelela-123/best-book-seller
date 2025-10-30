## Best Seller Prediction with `Logistic Regression` and Text Vectorizers

### 📘 Project Overview

This project implements a **Best Seller Prediction System** trained on book descriptions and ratings from a large Goodreads dataset. The model classifies books as **best sellers** (`1`) or **non-best sellers** (`0`) using **Logistic Regression** in combination with various **text vectorizers** (`CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`) in **scikit-learn**.

The notebook demonstrates a complete, **reproducible machine learning pipeline** from data preprocessing, feature extraction, and label creation, to model training, hyperparameter tuning, evaluation, and inference. The processed dataset and trained pipelines can be used to predict the likelihood of a book becoming a best seller based on its description and rating.

Dataset used:  
📘 [**Goodreads Book Datasets With User Rating 2M**](https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m)  
File used: **`book4000k-5000k.csv`**

---

### Dataset Description

| Column      | Type   | Description                                        |
| :---------- | :----- | :------------------------------------------------- |
| Description | Text   | Book synopsis or summary                           |
| Rating      | Float  | Average user rating of the book                    |
| BestSeller  | Binary | Label indicating whether the book is a best seller |

---

### 3. Preprocessing Pipeline

| Step               | Description                                                                |
| :----------------- | :------------------------------------------------------------------------- |
| Null Value Removal | Dropped rows with missing values                                           |
| Label Creation     | Books with rating > 2.5 labeled as best sellers (`1`), others as non (`0`) |
| Text Cleaning      | Removed punctuation, numbers, and extra spaces                             |
| Feature Extraction | Tokenization + Vectorization (`Count`, `TF-IDF`, `Hashing`)                |
| Train-Test Split   | Dataset split into training and testing sets                               |

### 4. Model Architecture

**Class**: LogisticRegression
| Component | Description |
| :------------------ | :-------------------------------------------------------- |
| Vectorizer | Converts text into numerical features (Count/TF-IDF/Hash) |
| Logistic Regression | Predicts the binary label (`BestSeller`) |

### 5. Evaluation Metrics

We evaluate the best model using:

1. **Accuracy**: Overall correctness of predictions
2. **F1-Score**: Balances precision and recall for imbalanced data
3. **Confusion Matrix**: Visualizes true positives, false positives, etc.
4. **Classification Report**: Detailed precision, recall, F1 for each class

### 🧠 Key Takeaways

- **Text vectorization** with n-grams and TF-IDF improves predictive performance.

- **Logistic Regression** performs well for binary classification of textual data.

- **Handling class imbalance** with `class_weight='balanced'` improves recall for the minority class.

- The pipeline can be reused for new book descriptions to predict potential best sellers.
-

### Inference Example

Use the trained pipeline to predict new book descriptions:

```shell
Enter the book description or (q) to quit: Playwright Power is a concise handbook on how to write a play. Written by an award-winning playwright, this book is built on proven basics. It provides important information for the beginning playwright as well as solid reinforcement for those already established. Beginning with the definition of a play, this book goes on to explain the components of the script, the playwright's best environment and some questions the new playwright should think about before beginning. This book guides the playwright through the plot, building characters, the dialogue, and how to get the play produced.
==================================================
📑 BOOK DESCRIPTION
==================================================
Playwright Power is a concise handbook on how to write a play. Written by an award-winning playwright, this book is built on proven basics. It provides important information for the beginning playwright as well as solid reinforcement for those already established. Beginning with the definition of a play, this book goes on to explain the components of the script, the playwright's best environment and some questions the new playwright should think about before beginning. This book guides the playwright through the plot, building characters, the dialogue, and how to get the play produced.

==================================================
🔮 BEST SELLER PREDICTION
==================================================
 PREDICTION              : 1
 BEST SELLER OUTCOME     : Yes
```

### Notebook

The notebook for training the model can be found [here - `00_Author_Best_Seller.ipynb`](/notebooks/00_Author_Best_Seller.ipynb).

### Report

The report for this task can be found in the `Word Document` that is located in the folder [docs](/docs/).
