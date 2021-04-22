import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = pickle.load(open('pretrained/my_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    features_list = [float(x) for x in request.form.values()]
    features = np.array(features_list).reshape(1,-1)
    predict_outcome_list = model.predict(features)
    predict_outcome = round(predict_outcome_list[0],2)

    return render_template('page.html',prediction_display_area='Predicted value：{}'.format(predict_outcome))

if __name__ == "__main__":
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()