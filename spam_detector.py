import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ============================
# STEP 0: Setup NLTK Resources
# ============================
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Lowercase, remove punctuation/numbers, tokenize, remove stopwords"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only alphabets
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ============================
# STEP 1: Load Dataset
# ============================
df = pd.read_csv("data/spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']

print("Sample data:")
print(df.head())

# ============================
# STEP 2: Preprocess Messages
# ============================
df['message_cleaned'] = df['message'].apply(preprocess_text)

# ============================
# STEP 3: Train-Test Split
# ============================
X = df['message_cleaned']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# STEP 4: Vectorization (TF-IDF)
# ============================
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ============================
# STEP 5: Train Model
# ============================
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ============================
# STEP 6: Evaluate Model
# ============================
y_pred = model.predict(X_test_tfidf)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================
# STEP 7: Try Custom Messages
# ============================
while True:
    msg = input("\nEnter a message to classify (or type 'exit' to quit): ")
    if msg.lower() == "exit":
        break
    msg_cleaned = preprocess_text(msg)
    msg_tfidf = vectorizer.transform([msg_cleaned])
    print("Prediction:", model.predict(msg_tfidf)[0])
