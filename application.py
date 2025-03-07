from flask import Flask,redirect,url_for,render_template, request
from src.pipeline.inference_pipeline import Inference
import numpy as np

application= Flask(__name__)

app = application

model = Inference()

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/jigme",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
            gender=request.form.get('gender')
            race_ethnicity=request.form.get('ethnicity')
            parental_level_of_education=request.form.get('parental_level_of_education')
            lunch=request.form.get('lunch')
            test_preparation_course=request.form.get('test_preparation_course')
            reading_score=float(request.form.get('writing_score'))
            writing_score=float(request.form.get('reading_score'))
            data = [gender,race_ethnicity,parental_level_of_education,
                    lunch,test_preparation_course,reading_score,writing_score,
                    ]
            features = model.Custom_data(data)
            math_score = model.predict(features)
            if math_score[0]>100:
                math_score=100
            else:
                math_score = math_score[0]
            return render_template("index.html",math_score =math_score)
if __name__ == "__main__":
    app.run(host="0.0.0.0")