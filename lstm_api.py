from flask import Flask, request
import numpy as np
from keras.models import Sequential,load_model



app = Flask(__name__)

model = load_model("checkpoint/lstm.h5")


@app.route('/predict', methods = ['GET'])
def api_predict():
	totalseqs = 128*3
	data = request.form['data']
	
	points = []
	bhv = data.split(',')
	inst = np.zeros(totalseqs)
	one = np.array([float(i) for i in bhv])
	if len(one) <= totalseqs:
	    inst[:len(one)] = one
	    
	else:
	    inst = one[:totalseqs]
	    
	inst = np.reshape(inst,[128,3])
	
	points=[inst]
	
	points = np.array(points)
	
	result = model.predict(points,batch_size=1,verbose=1)
	print(result)
	return str(result[0][0])



if __name__ == '__main__':
	app.run()