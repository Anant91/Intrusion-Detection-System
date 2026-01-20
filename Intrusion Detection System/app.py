from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ================= FLASK APP =================
app = Flask(__name__)

# ================= LOAD DATA =================
# Create sample dataset for demonstration
# Replace with actual NSL-KDD dataset when available
np.random.seed(42)
n_samples = 1000
n_features = 41

X_data = np.random.randn(n_samples, n_features)
y_data = np.random.randint(0, 2, n_samples)

data = pd.DataFrame(X_data, columns=[f"feature_{i}" for i in range(n_features)])
data["label"] = y_data

# Encode categorical features
categorical_cols = data.select_dtypes(include=["object"]).columns
feature_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    feature_encoders[col] = le

X = data.drop("label", axis=1)
y = data["label"]

# ================= TRAIN MODEL =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

FEATURE_COUNT = X.shape[1]

# ================= FRONTEND HTML =================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Intrusion Detection System</title>
    <style>
        body {
            font-family: Arial;
            background: #0f172a;
            color: white;
        }
        .container {
            width: 500px;
            margin: 100px auto;
            background: #1e293b;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            border-radius: 6px;
            margin-top: 10px;
        }
        button {
            margin-top: 15px;
            padding: 10px 20px;
            background: #22c55e;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        }
        #result {
            margin-top: 20px;
            font-size: 20px;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>🔐 Intrusion Detection System</h1>
    <p>Enter {FEATURE_COUNT} comma-separated network features</p>

    <textarea id="features"
    placeholder="Example: 0,491,2,1,0,..."></textarea>

    <button onclick="predict()">Check Traffic</button>
    <div id="result"></div>
</div>

<script>
function predict() {
    let input = document.getElementById("features").value;
    let features = input.split(",").map(Number);

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({features: features})
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerHTML = data.result;
    });
}
</script>

</body>
</html>
"""

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template_string(HTML_PAGE.replace("{FEATURE_COUNT}", str(FEATURE_COUNT)))

@app.route("/predict", methods=["POST"])
def predict():
    features = request.json.get("features")

    if len(features) != FEATURE_COUNT:
        return jsonify({
            "result": f"❌ Error: Expected {FEATURE_COUNT} features, got {len(features)}"
        })

    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    result = "✅ Normal Traffic" if prediction == 0 else "🚨 Intrusion Detected"
    return jsonify({"result": result})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
