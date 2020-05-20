import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,static_folder='./static/')
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if(prediction==0):
        output="No"
    else:
        output="Yes"

    return render_template('index.html', prediction_text='Prediction of Chronic kidney disease: {}'.format(output))

@app.route('/runModel',methods=['POST','GET'])
def runModel():
    data=request.get_json()['fields']
    result=dt.predict([data])
    print(result)
    return str(result).split('[')[1].split(']')[0]
if __name__ == "__main__":
    app.run(debug=True)
