from flask import Flask, request
import numpy as np
from keras.models import Sequential,load_model



app = Flask(__name__)

model = load_model("checkpoint/lstm.h5")
@app.route('/predict', methods = ['GET'])
def api_message():
	totalseqs = 128*3
	data = request.form['data']
	print(data)
	points = []
	bhv = data.split(',')[6:]
	inst = np.zeros(totalseqs)
	one = np.array([float(i) for i in bhv])
	if len(one) <= totalseqs:
	    inst[:len(one)] = one
	    
	else:
	    inst = one[:totalseqs]
	    
	inst = np.reshape(inst,[128,3])
	points.append(inst)
	points.append(inst)
	points = np.array(points)
	print(points)
	predict = model.predict(points,batch_size=1,verbose=1)
	print(predict)
	return predict



if __name__ == '__main__':
    app.run(debug=True)