from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open('model/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    age = int(request.form['age'])

    input_data = pd.DataFrame([{
        'area_sqft': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age
    }])

    prediction = model.predict(input_data)[0]

    return render_template(
        'index.html',
        prediction_text=f"Estimated House Price: ${prediction:,.0f}"
    )

if __name__ == '__main__':
    app.run(debug=True)
