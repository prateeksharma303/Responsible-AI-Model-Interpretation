import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load Dataset
df = pd.read_csv("emails.csv", encoding="latin-1")

df = df[['spam', 'text']]
df.columns = ['label', 'message']

# Preprocessing
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(df["message"])
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

feature_names = cv.get_feature_names_out()

spam_scores = model.feature_log_prob_[1]

top_spam_words = sorted(
    zip(feature_names, spam_scores),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("Top 10 Words Associated with Spam:\n")

for word, score in top_spam_words:
    print(word)