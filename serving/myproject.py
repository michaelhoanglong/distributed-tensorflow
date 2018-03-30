import time

from argparse import ArgumentParser

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf
import numpy as np
import os

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.examples.tutorials.mnist import input_data

from flask import Flask
app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def index():
	#host, port, image = parse_args()
	try:
		host = "localhost"
		port = 9000
		channel = implementations.insecure_channel(host, int(port))
		stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

		# dataset = input_data.read_data_sets('/tmp/Mnist', one_hot=True)
		# batch = dataset.train.next_batch(1)
		# print(batch[1])
		if(request.method == 'GET'):
			img = [[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
			numpyarray = np.array(img, np.float32)
			print("Expected result: %d" % (2))
		if(request.method == 'POST'):
			imgurl = request.form['url']
			os.system("wget -O /home/ubuntu/serveimg/img.jpg " + imgurl)
			img = cv2.imread('/home/ubuntu/serveimg/img.jpg')
			imgarray = []
			for i in range(0, len(img)):
			    for j in range(0, len(img[i])):
			        tmp = img[i][j]
			        px = 0
			        for item in tmp:
			            if(item > 5):
			                px = 1
			                break
			        imgarray.append(px)
			numpyarray = np.array(imgarray, np.float32)

		start = time.time()

		servingrequest = predict_pb2.PredictRequest()
		
		servingrequest.model_spec.name = 'mnist'
	    
		servingrequest.model_spec.signature_name = 'predict_images'

		servingrequest.inputs['images'].CopyFrom(make_tensor_proto(numpyarray, shape=[1, 784]))
		
		result = stub.Predict(servingrequest, 60.0)  # 60 secs timeout

	    
		end = time.time()
	    
		time_diff = end - start

		print(result.outputs['scores'].float_val)
		print(result)
	    
		print('time elapased: {}'.format(time_diff))

		result_list = result.outputs['scores'].float_val

		max_val = max(result_list)

		num_result = 0
		for i in range(0,len(result_list)):
			if(result_list[i] == max_val):
				num_result = i
		
		return str(num_result)
	except Exception as e:
		with open('/home/ubuntu/myproject/log.txt', 'a+') as f:
			error = "\n\n Internal Server Error \n" + str(e)
			f.write(error)
			f.close()

if __name__ == "__main__":
    app.run(host='0.0.0.0')
