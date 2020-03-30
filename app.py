import numpy as np
from flask import Flask, request, jsonify, render_template,url_for
import pickle
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
import keras
import io

app = Flask(__name__)

def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	keras.backend.get_session().run(tf.local_variables_initializer())
	return auc

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	regressor = load_model('gru.h5')
	i = [int(x) for x in request.form.values()]
	def predict(coef, history):
		yhat = 0.0
		for i in range(1, len(coef)+1):
			yhat += coef[i-1] * history[-i]
		return yhat
 
	def difference(dataset):
		diff = list()
		for i in range(1, len(dataset)):
			value = dataset[i] - dataset[i - 1]
			diff.append(value)
		return np.array(diff)


	df = pd.read_csv("data.csv")
	prices = df['Open'].values
	history = [x for x in prices]
	output = list()
	for t in range(i[0]):
		model = ARIMA(history, order=(4,2,1))
		model_fit = model.fit(trend='nc', disp=False)
		ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
		resid = model_fit.resid
		diff = difference(history)
		yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
		output.append(yhat)
		history.append(yhat)
	time_st = np.append(prices[len(prices)-119:],yhat)
	time_st = time_st.reshape(-1,1)
	sc = MinMaxScaler(feature_range = (0,1))
	scaled_set = sc.fit_transform(time_st)
	future = []
	for i in range(60,len(time_st)):
		future.append(time_st[i-60:i])
	future = np.array(future)
	nm = regressor.predict(future)
	predict_1 = sc.inverse_transform(nm)
    
	plt.plot(predict_1,color = 'blue', label = 'Predicted Stock Prices')
	plt.title("Stock Price Prediction")
	plt.xlabel('Time')
	plt.ylabel('Open Price')
	plt.legend()
	plt.savefig('static/trend.svg')
	img_url = url_for('static',filename = 'trend.svg')

	return render_template('output.html', prediction_text='Next price would be ₹ {}'.format(output[0]),user_image = img_url )



if __name__ == "__main__":
	app.run(debug=True)
