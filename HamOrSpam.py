import tkinter as tk
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_csv(r'./spam_ham_dataset.csv')

stop_words = set(stopwords.words("english"))
Lemmatizer = WordNetLemmatizer()


def preprocess_news(text):
    text = text.lower()
    tokens = word_tokenize(text)
    words_no_stop = [word for word in tokens if word not in stop_words]
    words_no_punct = [word for word in words_no_stop if word.isalnum() and word]
    lemmatized_words = [Lemmatizer.lemmatize(word) for word in words_no_punct]
    return ' '.join(lemmatized_words)


df['text'] = df['text'].apply(preprocess_news)

X = df['text']
y = df['label']

log_reg_model = LogisticRegression()
svm_model = SVC()
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=43, stratify=y)


tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


def fit_and_evaluate_model_lr(accuracies):
    log_reg_model.fit(X_train_tfidf, y_train)
    y_pred_lr = log_reg_model.predict(X_test_tfidf)

    models = {
        "Logistic Regression": (y_pred_lr, y_test)
    }

    for model_name, (y_pred, y_true) in models.items():
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='spam')
        recall = recall_score(y_true, y_pred, pos_label='spam')
        confusion_mat = confusion_matrix(y_true, y_pred)

        accuracies[model_name] = accuracy

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("Confusion Matrix:")
        print(confusion_mat)
        print()

    return tfidf_vectorizer, accuracies

def fit_and_evaluate_model_svm(accuracies):
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)

    models = {
        "SVM": (y_pred_svm, y_test)
    }

    for model_name, (y_pred, y_true) in models.items():
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='spam')
        recall = recall_score(y_true, y_pred, pos_label='spam')
        confusion_mat = confusion_matrix(y_true, y_pred)

        accuracies[model_name] = accuracy

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("Confusion Matrix:")
        print(confusion_mat)

        print()

    return tfidf_vectorizer, accuracies


def fit_and_evaluate_model_dt(accuracies):
    dt_model.fit(X_train_tfidf, y_train)
    y_pred_dt = dt_model.predict(X_test_tfidf)

    models = {
        "Decision Tree": (y_pred_dt, y_test)
    }

    for model_name, (y_pred, y_true) in models.items():
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='spam')
        recall = recall_score(y_true, y_pred, pos_label='spam')
        confusion_mat = confusion_matrix(y_true, y_pred)

        accuracies[model_name] = accuracy

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("Confusion Matrix:")
        print(confusion_mat)
        print()

    return tfidf_vectorizer, accuracies

def fit_and_evaluate_model_rf(accuracies):
    rf_model.fit(X_train_tfidf, y_train)
    y_pred_rf = rf_model.predict(X_test_tfidf)

    models = {
        "Random Forest": (y_pred_rf, y_test)
    }

    for model_name, (y_pred, y_true) in models.items():
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='spam')
        recall = recall_score(y_true, y_pred, pos_label='spam')
        confusion_mat = confusion_matrix(y_true, y_pred)


        accuracies[model_name] = accuracy

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("Confusion Matrix:")
        print(confusion_mat)
        print()

    return tfidf_vectorizer, accuracies


accuracies = {}
tfidf_vectorizer, accuracies = fit_and_evaluate_model_lr(accuracies)
tfidf_vectorizer, accuracies = fit_and_evaluate_model_svm(accuracies)
tfidf_vectorizer, accuracies = fit_and_evaluate_model_dt(accuracies)
tfidf_vectorizer, accuracies = fit_and_evaluate_model_rf(accuracies)
print("Final Accuracies:", accuracies)



def classify_text(text, model, tfidf_vectorizer):
    preprocessed_text = preprocess_news(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_tfidf)
    return prediction[0]


def classify_button_clicked(model):
    text = text_entry.get("1.0", tk.END)
    prediction = classify_text(text, model, tfidf_vectorizer)
    result_label.config(text=f"The text is classified as '{prediction}'.")


root = tk.Tk()
root.title("Spam/Ham Classifier")

text_entry = tk.Text(root, height=10, width=40)
text_entry.pack(pady=10)


models = [("Logistic Regression", log_reg_model), ("SVM", svm_model), ("Decision Tree", dt_model),
          ("Random Forest", rf_model)]
for model_name, model in models:
    ttk.Button(root, text=f"Classify with {model_name}",
               command=lambda model=model: classify_button_clicked(model)).pack()


def display_accuracy(model_name):
    accuracy = accuracies[model_name]
    accuracy_label.config(text=f"Accuracy of {model_name}: {accuracy}")


accuracy_label = tk.Label(root, text="")
accuracy_label.pack(pady=10)

for model_name in accuracies:
    ttk.Button(root, text=f"Display Accuracy of {model_name}",
               command=lambda model_name=model_name: display_accuracy(model_name)).pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)
root.mainloop()