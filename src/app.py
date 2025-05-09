from utils import db_connect
engine = db_connect()

"""
Sistema de Detección de URLs Spam usando SVM y NLP
"""

# 1. Importación de librerías
import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 2. Descarga de recursos necesarios para preprocesamiento de texto
download("stopwords")
download("wordnet")

# 3. Cargar el dataset directamente desde GitHub
df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv")
print("Dataset cargado con éxito:", df.shape)

# 4. Preprocesamiento de URLs
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_url(url):
    url = url.lower()
    url = re.sub(r"[\\/\\.\\-_=:\\?&%]+", " ", url)
    url = re.sub(r"[^a-z ]+", "", url)
    tokens = url.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in stop_words and len(tok) > 2]
    return " ".join(tokens)

df["tokens"] = df["url"].apply(preprocess_url)
print("Preprocesamiento completado.")

# 5. Vectorización con TF-IDF
corpus = df["tokens"].values
vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=3)
X = vectorizer.fit_transform(corpus).toarray()
y = df["is_spam"].astype(int)

# 6. División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Datos divididos: Train =", X_train.shape, "| Test =", X_test.shape)

# 7. Optimización de hiperparámetros con Grid Search
svc = SVC()
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 8. Evaluación del mejor modelo
best_model = grid_search.best_estimator_
y_pred_opt = best_model.predict(X_test)

print("\n Mejores parámetros encontrados:", grid_search.best_params_)
print(" Accuracy:", accuracy_score(y_test, y_pred_opt))
print("\n Reporte de Clasificación:\n", classification_report(y_test, y_pred_opt))
print("\n Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_opt))

# 9. Guardar el modelo y el vectorizador
with open("svm_url_spam_rbf_c10.sav", "wb") as f:
    pickle.dump(best_model, f)

with open("tfidf_vectorizer.sav", "wb") as f:
    pickle.dump(vectorizer, f)

print(" Modelo y vectorizador guardados correctamente.")
