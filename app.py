import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def price_prediction():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
                    origin = request.form.get('origin'),
                    destination = request.form.get('destination'),
                    start_date = request.form.get('start_date'),
                    end_date = request.form.get('end_date'),
                    train_type = request.form.get('train_type'),
                    train_class = request.form.get('train_class'),
                    fare = request.form.get('fare')
        )

        prediction = data.get_data_as_frame()
        print(prediction)
        print("Before prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        results = predict_pipeline.predict(prediction)
        print("After prediction")

        return render_template('index.html', results = results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)