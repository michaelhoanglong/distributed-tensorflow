import time

from argparse import ArgumentParser

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf
import numpy as np
import pngtopixel as imgtopx
import cv2

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.examples.tutorials.mnist import input_data



def parse_args():
    parser = ArgumentParser(description="Request a TensorFlow server for a prediction on the image")
    parser.add_argument("-s", "--server",
                        dest="server",
                        default='172.17.0.2:2000',
                        help="prediction service host:port")
    parser.add_argument("-i", "--image",
                        dest="image",
                        default="",
                        help="path to image in JPEG format",)
    args = parser.parse_args()

    host, port = args.server.split(':')
    
    return host, port, args.image


def main():
    # parse command line arguments
    host, port, image = parse_args()

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # dataset = input_data.read_data_sets('/tmp/Mnist', one_hot=True)
    # batch = dataset.train.next_batch(1)
    # print(batch[1])
    #img = [[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    #numpyarray = imgtopx.imageprepare(image)
    img = cv2.imread(image)
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

    request = predict_pb2.PredictRequest()

    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'

    request.inputs['images'].CopyFrom(make_tensor_proto(numpyarray, shape=[1, 784]))
    # request.inputs['keep_prob'].CopyFrom(make_tensor_proto(1, shape=[1,1]))

    result = stub.Predict(request, 60.0)  # 60 secs timeout

    end = time.time()
    time_diff = end - start

    #print(result.outputs['scores'].float_val)
    print(result)
    print('time elapased: {}'.format(time_diff))

    result_list = result.outputs['scores'].float_val

    max_val = max(result_list)

    num_result = 0
    for i in range(0,len(result_list)):
        if(result_list[i] == max_val):
            num_result = i
    print('Predicted Number: {}'.format(num_result))


if __name__ == '__main__':
    main()
