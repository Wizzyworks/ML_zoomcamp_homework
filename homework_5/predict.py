import pickle 

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

dv, model

app = Flask('lead')
@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    lead = y_pred >=0.5

    result = {
        'lead_prob': float(y_pred),
        'lead': bool(lead)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

