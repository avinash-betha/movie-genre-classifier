from flask import Flask, request, render_template
from src.predict import predict_genre

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    plot = request.form['plot']
    if plot.strip() == "":
        return render_template('index.html', prediction_text="Please enter a valid plot.")
    
    genre, confidence = predict_genre(plot)
    result_text = f'Predicted Genre: <strong>{genre}</strong><br>Confidence: <strong>{confidence * 100:.2f}%</strong>'
    return render_template('index.html', prediction_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)
