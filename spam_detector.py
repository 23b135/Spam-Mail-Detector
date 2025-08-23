import pandas as pd
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# ============================
# STEP 1: Load Dataset
# ============================
df = pd.read_csv("data/spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']

print("Sample data:")
print(df.head())

# ============================
# STEP 2: Preprocessing
# ============================
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = word_tokenize(text)  # tokenize
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    return " ".join(tokens)

df['cleaned_message'] = df['message'].apply(preprocess_text)

# ============================
# STEP 3: Train-Test Split
# ============================
X = df['cleaned_message']
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
    prediction = model.predict(msg_tfidf)[0]
    
    if prediction == "spam":
        print("Prediction: SPAM (🚨 Junk/Unwanted Message)")
    else:
        print("Prediction: HAM (✅ Legitimate Message)")
