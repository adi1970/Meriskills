from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load("pipe.pkl")

@app.route("/predict", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        bmi = float(request.form["bmi"])
        age = float(request.form["age"])
        pregnancies = float(request.form["pregnancies"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        diabetes_pedigree = float(request.form["diabetes_pedigree"])

        # Prepare the input features for prediction
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
        
        # Make a prediction
        prediction = model.predict(input_data)

        # Display the prediction result
        return render_template("prediction.html", prediction=prediction[0])

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=7000)
