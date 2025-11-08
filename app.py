from flask import Flask, render_template, request
import pandas as pd
import os
from ml.predict_attack import predict_single, predict_batch  # We'll adapt your existing code

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file is uploaded
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Load CSV and predict
            df = pd.read_csv(filepath)
            predictions = predict_batch(df)  # returns list/series of predictions
            
            df["Prediction"] = predictions
            return render_template("results.html", tables=[df.to_html(classes="data", index=False)])
        
        # Single input prediction from form
        elif request.form:
            input_data = {key: float(value) for key, value in request.form.items()}
            pred = predict_single(input_data)
            return render_template("results.html", prediction=pred, single=True)
    
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
