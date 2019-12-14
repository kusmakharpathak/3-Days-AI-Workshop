import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	from tensorflow.keras.models import model_from_json


def init():
	json_file = open('model/mnist.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load weights into new model
	loaded_model.load_weights("model/mnist.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	return loaded_model
