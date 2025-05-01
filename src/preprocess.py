import pandas as pd
import re

def clean_text(text):
    ## Lowercase, remove punctuation and numbers.
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def load_and_clean_data(path):
    ### Load CSV, clean plot column, simplify genres, and return filtered DataFrame.
    df = pd.read_csv(path)
    
    # Ensure required columns are present
    if 'overview' not in df.columns or 'genres' not in df.columns:
        raise ValueError("Dataset must contain 'overview' and 'genres' columns.")

    # Simplify genres to the first genre only (single-label)
    df['genres'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')

    # Remove rows with missing/duplicate values
    df = df[['overview', 'genres']].dropna().drop_duplicates()
    df.rename(columns={'overview': 'plot', 'genres': 'genre'}, inplace=True)

    # Filter out rare genres (keep only genres that appear more than 20 times)
    genre_counts = df['genre'].value_counts()
    valid_genres = genre_counts[genre_counts > 20].index.tolist()
    df = df[df['genre'].isin(valid_genres)]

    # Clean the plot text
    df['plot_clean'] = df['plot'].apply(clean_text)
    
    return df