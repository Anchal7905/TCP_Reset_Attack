"""
app.py
Flask web app for TCP Reset Attack detection using ML model.
Supports single and batch predictions with dynamic CSV mapping and preview.
"""

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Import model logic
from ml.predict_attack import predict_single, predict_batch, REQUIRED_FEATURES

app = Flask(__name__)
app.secret_key = "secret123"

# Upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html", features=REQUIRED_FEATURES)


# ---------- SINGLE PREDICTION ----------
@app.route("/predict_single", methods=["POST"])
def single_predict():
    try:
        flow_features = {}
        for feature in REQUIRED_FEATURES:
            flow_features[feature] = float(request.form.get(feature, 0))
        prediction = predict_single(flow_features)
        return render_template("index.html", features=REQUIRED_FEATURES, single_result=prediction)
    except Exception as e:
        flash(f"Error during single prediction: {e}", "error")
        return redirect(url_for("index"))


# ---------- CSV PREVIEW ----------
@app.route("/preview_csv", methods=["POST"])
def preview_csv():
    try:
        file = request.files["file"]
        if not file:
            flash("No file uploaded!", "error")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        columns = list(df.columns)
        # Save path for next step
        return render_template(
            "index.html",
            features=REQUIRED_FEATURES,
            columns=columns,
            file_path=filepath,
            preview=df.head().to_html(classes="table table-bordered table-striped")
        )
    except Exception as e:
        flash(f"Error previewing file: {e}", "error")
        return redirect(url_for("index"))


# ---------- BATCH PREDICTION ----------
@app.route("/predict_batch", methods=["POST"])
def batch_predict():
    try:
        file_path = request.form.get("file_path")
        if not file_path or not os.path.exists(file_path):
            flash("Please upload and preview the CSV first!", "error")
            return redirect(url_for("index"))

        predictions = predict_batch(file_path)
        df = pd.read_csv(file_path)
        df["Prediction"] = predictions

        # Show top 10 for clarity
        preview_html = df.head(10).to_html(classes="table table-bordered table-striped")

        return render_template(
            "index.html",
            features=REQUIRED_FEATURES,
            batch_result_preview=preview_html,
            success="Batch prediction completed successfully!"
        )
    except Exception as e:
        flash(f"Error processing batch: {e}", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
