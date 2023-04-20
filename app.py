from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
from prediction import Prediction, Preparation
from prediction import vis
import pandas as pd

app = Flask(__name__, template_folder='./webapp/src/app/home/')
CORS(app)

pr = Prediction()

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    if not file or (file.content_type != 'text/csv' and file.content_type != 'application/vnd.ms-excel' and file.content_type != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
        return jsonify({'message': 'Invalid file type.'}), 400
    
    df = pd.read_csv(file)
    df.to_csv('input.csv', index=False)
    
    M = request.form.get('selectedNumerix')
    N = request.form.get('selectedPeriodicity')
    
    pr.prepare()
    pr.train()

    res = pr.load_predict(M, N)
    res = list(res)
    out = "Success "
    for i in res:
      out += str(i) + " "
    print("Prediction Successful", res)
    
    out = pd.read_csv('./output.csv')
    vis(out)
    return 'Success'

if __name__ == '__main__':
    app.run(debug=True)
