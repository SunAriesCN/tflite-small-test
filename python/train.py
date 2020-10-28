# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import shutil
import argparse

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def parser():
    parser = argparse.ArgumentParser('Train the Comparator.')
    parser.add_argument('--data_dir', '-d', default=None)
    parser.add_argument('--label_file_path', '-l', default=None)
    parser.add_argument('--batch_size', '-b', default=1)
    parser.add_argument('--learning_rate', '-lr', default=1e-2) 
    parser.add_argument('--epochs', '-e', default=25)
    parser.add_argument('--output_dir', '-o', default='models')

    return parser.parse_args()


def ranking_loss(y_true, y_pred):
    margin = 1.
    element_product = tf.math.multiply(y_true, y_pred, name='element_product')
    loss = tf.keras.maximum(margin - element_product, 0)
    return tf.keras.mean(loss)


def decode_labels_to_list(train_data_dir, label_path):

    if not os.path.exists(label_path):
        return None

    reading_labels = []
    
    with open(label_path, mode='r') as label_file:
        for line in label_file:
            info = line[:-1].split(',')

            left_image_name = os.path.basename(info[0])
            left_imag_path = os.path.join(train_data_dir, left_image_name)
            
            right_image_name = os.path.basename(info[1])
            right_image_path = os.path.join(train_data_dir, right_image_name)

            label = info[-1]

            reading_labels.append((left_imag_path, right_image_path, label))

    return reading_labels


def read_image_to_buffer(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.io.decode_image(image, channels=3)
    image = image[..., ::-1]
    return image


def process_fn(image_pair_and_label):

    first_image_path = image_pair_and_label[0]
    first_image = read_image_to_buffer(first_image_path)

    second_image_path = image_pair_and_label[1]
    second_image = read_image_to_buffer(second_image_path)   

    pairs = [first_image, second_image] 
    
    labels = float(image_pair_and_label[-1])
    print(pairs)
    print(labels)
    return pairs, labels

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

class Trainer:
    def __init__(self, train_data_dir, label_file_path, batch_size):

        assert os.path.exists(train_data_dir), f"Training directory {train_data_dir} is not exists."
        assert os.path.exists(label_file_path), f"Label fiel {label_file_path} is not exists."

        # Process training data and label file path
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        image_pairs_and_labels = decode_labels_to_list(train_data_dir, label_file_path)

        total_samples_size = len(image_pairs_and_labels)
        train_samples_size = int(total_samples_size*0.9)

        train_image_pairs_and_labels = image_pairs_and_labels[:train_samples_size]
        train_dataset = tf.data.Dataset.from_tensor_slices(train_image_pairs_and_labels)
        train_dataset = train_dataset.map(process_fn, num_parallel_calls=AUTOTUNE)
        # Can disable following test codes with comments
        for i, data in enumerate(train_dataset):
            cv2.imshow("aaaa",data[0][0].numpy())
            cv2.waitKey()
        train_dataset = train_dataset.shuffle(buffer_size=train_samples_size)
        train_dataset = train_dataset.batch(batch_size=batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

        self.train_dataset = train_dataset

        # Define data input preprocessing method
        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # Build compare model.
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=None,
            alpha=0.5,
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            pooling='avg'
        )

        self.base_model.trainable = False

        top_dropout_rate = 0.2
        x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(self.base_model.output)
        output = tf.keras.layers.Dense(1, activation='linear', name='prediction')(x)

        self.network = tf.keras.Model(self.base_model.input,
                                      output, name='TestComparator')

        print(f'self.base_model.input:{self.base_model.input}')
        print(f'self.network.input: {self.network.input}')
        # input_shape = [None, 2, None, None, 3]
        # network_input = tf.keras.Input(shape=input_shape)
        # print(f'one of network_input :{network_input[:][0]}')
        input_shape = [None, None, 3]
        first_input = tf.keras.Input(shape=input_shape)#, dtype="uint8")
        first_score = self.network(self.preprocess_input(first_input))

        second_input = tf.keras.Input(shape=input_shape)#, dtype="uint8")
        second_score = self.network(self.preprocess_input(second_input))

        print(f'first_feature:{first_score}')
        print(f'second_feature:{second_score}')
        subtracted = tf.keras.layers.Subtract()([first_score, second_score])
        # subtraction_layer = tf.keras.layers.Lambda(lambda tensors: tensors[0] - tensors[1])
        # subtracted = subtraction_layer([first_feature, second_feature])

        self.model = tf.keras.Model(inputs=[first_input, second_input],
                                    outputs=subtracted,
                                    name="JustForTraining")

        print(f"model input: {self.model.input}")
        print(f"model output: {self.model.output}")

        
    def run_train(self, output_dir, learning_rate=1e-2, epochs=25):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model.compile(optimizer=optimizer, loss=ranking_loss, metrics=ranking_loss)

        self.model.summary()

        history = self.model.fit(self.train_dataset,
                                 epochs=epochs,
                                 validation_data=None, verbose=2)

        plot_hist(history)

        output_path = os.path.join(output_dir, "abc.h5")
        self.model.save(output_path,
                        overwrite=True,
                        include_optimizer=True,
                        save_format=None,
                        signatures=None,
                        options=None)


def train(args):

    trainer = Trainer(train_data_dir = args.data_dir,
                      label_file_path = args.label_file_path,
                      batch_size = args.batch_size)

    trainer.run_train(output_dir=args.output_dir,
                      learning_rate=args.learning_rate, epochs=args.epochs)

if __name__ == '__main__':
    train(parser())

