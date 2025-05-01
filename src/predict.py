import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from src.preprocess import clean_text

def predict_genre(plot):
    # Load the trained model
    model = joblib.load('genre_classifier.pkl')

    # Clean the input plot
    cleaned = clean_text(plot)
    print("Cleaned Plot:", cleaned)

    # Get prediction probabilities
    probs = model.predict_proba([cleaned])[0]
    predicted_class = model.classes_[probs.argmax()]
    confidence = probs.max()

    print(f"Predicted Genre: {predicted_class} (Confidence: {confidence:.2f})")
    return predicted_class, confidence

# Manual test
if __name__ == "__main__":
    new_plot = "A man wakes up to find himself in a mysterious room, with no memory of how he got there."
    genre, conf = predict_genre(new_plot)
    print(f"Genre: {genre} (Confidence: {conf:.2f})")
