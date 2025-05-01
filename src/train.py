import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_clean_data

def train_model():
    # Load and clean data
    df = load_and_clean_data('data/movies.csv')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['plot_clean'], df['genre'], test_size=0.2, random_state=42
    )

    # Define pipeline with optimized TF-IDF
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=15000,         # increased feature space
            max_df=0.9,                 # ignore too frequent terms
            min_df=2,                   # ignore rare terms
            ngram_range=(1, 2)          # include unigrams and bigrams
        )),
        ('clf', LogisticRegression(
            max_iter=2000,              # allow more iterations for convergence
            class_weight='balanced',
            solver='lbfgs',             # more stable for multi-class
            multi_class='multinomial', # better for non-binary classification
            C=1.0                       # regularization strength (can tune)
        ))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(pipeline, 'genre_classifier.pkl')
    print("Model saved as 'genre_classifier.pkl'")

if __name__ == "__main__":
    train_model()
