import time

from argparse import ArgumentParser

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf
import numpy as np
import os
import subprocess
import cv2
import traceback
import thread
import time
import shutil

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.examples.tutorials.mnist import input_data

from flask import Flask, request
app = Flask(__name__)

#@app.after_request
#def after_request(response):
#    response.headers.add('Access-Control-Allow-Origin', '*')
#    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#    return response

def runServingService(modelName):
    modelNameParam = "--model_name=" + modelName
    try:
        killPreviousProcess = subprocess.check_call(["sudo", "killall", "-9", "tensorflow_model_server"])
    except Exception as e:
        print("pass exception")
    finally:
        runServing = subprocess.check_call(["tensorflow_model_server", "--port=9000", str(modelNameParam), "--model_base_path=/home/ubuntu/model"])


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
            #img = [[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
            numpyarray = np.array(img, np.float32)
	    print("Expected result: %d" % (2))
	if(request.method == 'POST'):
            modelUrl = request.form.get('modelUrl')
            imgUrl = request.form.get('imageUrl')
            modelName = request.form.get('modelName')
            imageSize = request.form.get('imageSize')
            checkClearModelPath = 1
            modelFolder = "/home/ubuntu/model"
            if(os.listdir(modelFolder) != []): 
                #checkClearModelPath = subprocess.check_call(["sudo", "rm", "-r", "/home/ubuntu/model/*"])  
                for the_file in os.listdir(modelFolder):
                    file_path = os.path.join(modelFolder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path): shutil.rmtree(file_path)
                    except Exception as e:
                        raise Exception(str(e))

            #if(checkClearModelPath == 0):
            downloadModel = subprocess.check_call(["wget", "-O", "/home/ubuntu/model/model.zip", modelUrl])
            if(downloadModel == 0):
                unzipModel = subprocess.check_call(["unzip", "/home/ubuntu/model/model.zip", "-d", "/home/ubuntu/model/"])
                if(unzipModel == 0):
                    try:
                        thread.start_new_thread(runServingService, (modelName,))
                    except Exception as e:
                        raise Exception(str(e))

            checkRunningProcess = ""
            while("tensorflow_model_server" not in checkRunningProcess):
                checkRunningProcess = subprocess.check_output(['ps', '-au'])
                print(checkRunningProcess)
            
            time.sleep(1.5)
            downloadImg = subprocess.check_call(["wget", "-O", "/home/ubuntu/serveimg/img.jpg", imgUrl])
            if(downloadImg == 0):
                print("downloaded image")
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
            else:
                raise Exception("cannot download image") 

        start = time.time()

	servingrequest = predict_pb2.PredictRequest()
		
	servingrequest.model_spec.name = modelName
	    
	servingrequest.model_spec.signature_name = 'predict_images'

	#servingrequest.inputs['images'].CopyFrom(make_tensor_proto(numpyarray, shape=[1, 784]))
        servingrequest.inputs['images'].CopyFrom(make_tensor_proto(numpyarray, shape=[1, imageSize]))

		
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
	    error = "\n\n Internal Server Error \n" + str(traceback.format_exc())
	    f.write(error)
	    f.close()

if __name__ == "__main__":
    app.debug = False
    app.run(host='0.0.0.0')
