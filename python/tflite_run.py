# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import argparse
import numpy as np
import tensorflow as tf

class TestCase:
    def __init__(self, model_path):
        # Load model to init interpreter
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        
        self._input_details = self._interpreter.get_input_details()
        print(f"input_details: {self._input_details}")
        
        input_size_height = self._input_details[0]['shape'][1]
        input_size_width = self._input_details[0]['shape'][2]
        self._input_size = (input_size_width, input_size_height)
    
        self._output_details = self._interpreter.get_output_details()
        print(f"output_details: {self._output_details}")


    def get_input_size(self):
        return self._input_size

    def run_model(self, input_buffer):

        assert not input_buffer is None

        print(input_buffer.shape)
        self._interpreter.resize_tensor_input(self._input_details[0]['index'], input_buffer.shape)
        self._interpreter.allocate_tensors()
        self._interpreter.set_tensor(self._input_details[0]['index'], input_buffer)
        self._interpreter.invoke()
        landmarks_buffer = self._interpreter.get_tensor(self._output_details[0]['index'])
        hand_presences_buffer = self._interpreter.get_tensor(self._output_details[1]['index'])
        handness_buffer = self._interpreter.get_tensor(self._output_details[2]['index'])
        return landmarks_buffer, hand_presences_buffer, handness_buffer


def parser():
    parser = argparse.ArgumentParser('Small test for TensorFlow Lite models.')

    parser.add_argument('--data', '-d', default=None, dest='data',
                        type=str, help='test data inputs')

    parser.add_argument('--model', '-m', dest='model', \
                        type=str, default=None, help='model test')

    return parser.parse_args()

def main():
    args = parser()

    assert os.path.exists(args.data), f'{args.data} is not exist'

    test_case = TestCase(args.model)  
    
    # Preprocess function
    def preprocess(image, size):
        resized_image = cv2.resize(image, size)
        return resized_image.astype(np.float32) / 255.

    # Image data preparation
    images = []
    input_size = test_case.get_input_size()
    for image_name in os.listdir(args.data):

        image_path = os.path.join(args.data, image_name)
        image = cv2.imread(image_path)
        assert not image is None, f'{image_path} is empty image'
        images.append(preprocess(image, input_size))

    # Batching run
    input_buffer = np.array(images)
    landmarks_buffer, hand_presences_buffer, handness_buffer \
        = test_case.run_model(input_buffer)

    print('Bathching Run:')
    print(f'landmarks: {landmarks_buffer}')
    print(f'handness: {handness_buffer}')
    print(f'hand_presences: {hand_presences_buffer}')  


    # Loop run
    landmarks_buffers = []
    hand_presences_buffers = []
    handness_buffers = []
    for image in images:
        input_buffer = np.array([image])
        landmarks_buffer, hand_presences_buffer, handness_buffer \
            = test_case.run_model(input_buffer)

        landmarks_buffers.append(landmarks_buffer)
        hand_presences_buffers.append(hand_presences_buffer)
        handness_buffers.append(handness_buffer)

    print('Loop Run:')
    print(f'landmarks: {np.concatenate(landmarks_buffers, axis=0)}')
    print(f'handness:{np.concatenate(handness_buffers, axis=0)}')
    print(f'hand_presences:{np.concatenate(hand_presences_buffers, axis=0)}')      


if __name__ == '__main__':
    main()
