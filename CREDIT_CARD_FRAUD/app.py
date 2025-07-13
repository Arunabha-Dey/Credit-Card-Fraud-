from flask import Flask, render_template, request
import pandas as pd
import joblib
from geopy.distance import geodesic

app = Flask(__name__)

# Load model and encoders
model = joblib.load("Fraud_Detection_Model.jb")
encoder = joblib.load("label_encoders.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        merchant = request.form["merchant"]
        category = request.form["category"]
        amt = float(request.form["amt"])
        lat = float(request.form["lat"])
        long = float(request.form["long"])
        merch_lat = float(request.form["merch_lat"])
        merch_long = float(request.form["merch_long"])
        hour = int(request.form["hour"])
        day = int(request.form["day"])
        month = int(request.form["month"])
        gender = request.form["gender"]
        cc_num = request.form["cc_num"]

        distance = haversine(lat, long, merch_lat, merch_long)

        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant','category','amt','distance','hour','day','month','gender','cc_num'])

        categorical_col = ['merchant','category','gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        prediction = model.predict(input_data)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
