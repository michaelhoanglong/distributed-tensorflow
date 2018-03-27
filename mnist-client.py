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



def parse_args():
    parser = ArgumentParser(description="Request a TensorFlow server for a prediction on the image")
    parser.add_argument("-s", "--server",
                        dest="server",
                        default='172.17.0.2:9000',
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

    # Send request
    # with open(image, 'rb') as f:
    #     # See prediction_service.proto for gRPC request/response details.
    #     data = f.read()

    #     start = time.time()

    #     request = predict_pb2.PredictRequest()

    #     # Call GAN model to make prediction on the image
    #     request.model_spec.name = 'gan'
    #     request.model_spec.signature_name = 'predict_images'
    #     request.inputs['images'].CopyFrom(make_tensor_proto(data, shape=[1]))

    #     result = stub.Predict(request, 60.0)  # 60 secs timeout

    #     end = time.time()
    #     time_diff = end - start

    #     print(result)
    #     print('time elapased: {}'.format(time_diff))
    dataset = input_data.read_data_sets('/tmp/Mnist', one_hot=True)
    batch = dataset.train.next_batch(1)
    print(batch[1])
    start = time.time()

    request = predict_pb2.PredictRequest()

    # Call GAN model to make prediction on the image
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'

    request.inputs['images'].CopyFrom(make_tensor_proto(batch[0], shape=[1, 784]))
    # request.inputs['keep_prob'].CopyFrom(make_tensor_proto(1, shape=[1,1]))

    result = stub.Predict(request, 60.0)  # 60 secs timeout

    end = time.time()
    time_diff = end - start

    print(result)
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    main()
