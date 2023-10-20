import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def price_prediction():
    if request.method == 'POST':
        # origin
        origin = request.form.get('origin')

        # destination
        destination = request.form.get('destination')

        # start date
        start_date = request.form.get('start_date')
        start_day = int(pd.to_datetime(start_date, format="%Y-%m-%dT%H:%M").day)
        start_minutes = int(pd.to_datetime(start_date, format="%Y-%m-%dT%H:%M").minute)
        start_hours = int(pd.to_datetime(start_date, format="%Y-%m-%dT%H:%M").hour)
        start_month = int(pd.to_datetime(start_date, format="%Y-%m-%dT%H:%M").month)
        
        # end date
        end_date = request.form.get('end_date')
        end_day = int(pd.to_datetime(end_date, format="%Y-%m-%dT%H:%M").day)
        end_minutes = int(pd.to_datetime(end_date, format="%Y-%m-%dT%H:%M").minute)
        end_hours = int(pd.to_datetime(end_date, format="%Y-%m-%dT%H:%M").hour)
        end_month = int(pd.to_datetime(end_date, format="%Y-%m-%dT%H:%M").month)

        # train type
        train_type = request.form.get('train_type')

        # train class
        train_class = request.form.get('train_class')

        # fare
        fare = request.form.get('fare')

        data = CustomData(
                origin = origin,
                destination = destination,
                start_day = start_day,
                start_minutes = start_minutes,
                start_hours = start_hours,
                start_month = start_month,
                end_day = end_day,
                end_hours = end_hours,
                end_minutes = end_minutes,
                end_month = end_month,
                train_type = train_type,
                train_class = train_class,    
                fare = fare
        )

        prediction = data.get_data_as_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(prediction)

        return render_template('index.html', results = results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)