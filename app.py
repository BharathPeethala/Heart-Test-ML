from copyreg import pickle
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
sc = pickle.load(open('sc.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    lst = [0]*4
    cp = int(request.form['chest pain type(4 values)'])
    lst[cp] = 1
    trestbps = int(request.form['resting blood pressure'])
    lst += [trestbps]
    chol = int(request.form['serum cholestoral in mg/dl'])
    lst += [chol]
    fbs = int(request.form["fasting blood sugar > 120 mg/dl"])
    if fbs == 0:
        lst += [1, 0]
    else:
        lst += [0, 1]
    restecg = int(
        request.form['resting electrocardiographic results (values 0,1,2)'])
    dum = [0]*3
    dum[restecg] = 1
    lst += dum
    thalach = int(request.form['maximum heart rate achieved'])
    lst += [thalach]
    exang = int(request.form['exercise induced angina'])
    if exang == 0:
        lst += [1, 0]
    else:
        lst += [0, 1]
    final_features = np.array([lst])
    pred = model.predict(sc.transform(final_features))
    return render_template('result.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)
