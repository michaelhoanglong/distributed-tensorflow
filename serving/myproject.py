import time

from argparse import ArgumentParser

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.examples.tutorials.mnist import input_data

from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
	#host, port, image = parse_args()
	host = "localhost"
	port = 9000

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

	dataset = input_data.read_data_sets('/tmp/Mnist', one_hot=True)
    batch = dataset.train.next_batch(1)
    print(batch[1])
    start = time.time()

    request = predict_pb2.PredictRequest()

    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'

    request.inputs['images'].CopyFrom(make_tensor_proto(batch[0], shape=[1, 784]))

    result = stub.Predict(request, 60.0)  # 60 secs timeout

    end = time.time()
    time_diff = end - start

    print(result)
    print('time elapased: {}'.format(time_diff))

    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0')