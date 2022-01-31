# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py
import sklearn
# import the necessary packages
import dill
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)
	print(model)

modelpath = "/app/app/models/logreg_pipeline.dill"
# modelpath = "/home/dg/PycharmProjects/new/models/logreg_pipeline.dill"
# modelpath = '/Users/dmitriigribanov/PycharmProjects/test_proj_ml/models/logreg_pipeline.dill'


load_model(modelpath)

@app.route("/", methods=["GET"])
def general():
	return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":

		Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease = "", "", "", "", "", "", "", "", "", "", "", ""
		request_json = flask.request.get_json()
		if request_json["Age"] is not None:
			Age = request_json['Age']

		if request_json["Sex"] is not None:
			Sex = request_json['Sex']

		if request_json["ChestPainType"] is not None:
			ChestPainType = request_json['ChestPainType']

		if request_json["RestingBP"] is not None:
			RestingBP = request_json['RestingBP']

		if request_json["Cholesterol"] is not None:
			Cholesterol = request_json['Cholesterol']

		if request_json["FastingBS"] is not None:
			FastingBS = request_json['FastingBS']

		if request_json["RestingECG"] is not None:
			RestingECG = request_json['RestingECG']

		if request_json["MaxHR"] is not None:
			MaxHR = request_json['MaxHR']

		if request_json["ExerciseAngina"] is not None:
			ExerciseAngina = request_json['ExerciseAngina']

		if request_json["Oldpeak"] is not None:
			Oldpeak = request_json['Oldpeak']

		if request_json["ST_Slope"] is not None:
			ST_Slope = request_json['ST_Slope']

		if request_json["HeartDisease"] is not None:
			HeartDisease = request_json['HeartDisease']
		logger.info(f'{dt} Data: Age={Age}, Sex={Sex}, ChestPainType={ChestPainType}')
		try:
			preds = model.predict_proba(pd.DataFrame({"Age": [Age],
													  "Sex": [Sex],
													  "ChestPainType": [ChestPainType],
													  "RestingBP": [RestingBP],
													  "Cholesterol": [Cholesterol],
													  "FastingBS": [FastingBS],
													  "RestingECG": [RestingECG],
													  "MaxHR": [MaxHR],
													  "ExerciseAngina": [ExerciseAngina],
													  "Oldpeak": [Oldpeak],
													  "ST_Slope": [ST_Slope],
													  "HeartDisease": [HeartDisease],})
										)
		except AttributeError as e:
			logger.warning(f'{dt} Exception: {str(e)}')
			data['predictions'] = str(e)
			data['success'] = False
			return flask.jsonify(data)

		data["predictions"] = preds[:, 1][0]
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
