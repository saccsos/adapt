'''testing neural networks.
'''
#from __future__ import annotations
from typing     import *

import os, sys
import argparse

import tensorflow as tf


def set_gpu() -> None:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    return


class Executer:
    
    def __init__(self
                 , target_model: str
                 , mode: str):
        self.target_model = target_model
        self.mode         = mode


    def _gen_dataset(self) -> None:
        '''Generate dataset, the format is as follow:
        X_train, X_test, y_train, y_test
        '''
        res = dict()
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        res['X_train'] = X_train
        res['X_test']  = X_test
        res['y_train'] = y_train
        res['y_test']  = y_test

        self.mnist = res
        print(f'[LOG]: minst dataset is generated.')
        return 


    def LeNet5(self):
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Convolution2D
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Input
        from tensorflow.keras.layers import MaxPooling2D
        from tensorflow.keras.models import Model
        
        # Input layer.
        input_tensor = Input(shape=(28, 28, 1))
        
        # Block 1.
        x = Convolution2D(6, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
        
        # Block 2.
        x = Convolution2D(16, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
        
        # Fully connected.
        x = Flatten(name='flatten')(x)
        x = Dense(120, activation='relu', name='fc1')(x)
        x = Dense(84, activation='relu', name='fc2')(x)
        x = Dense(10, name='before_softmax')(x)
        x = Activation('softmax', name='redictions')(x)
        
        print(f'[LOG]: LeNet5 is ready to train')
        return Model(input_tensor, x)


    def _gen_model(self) -> None:
        if self.target_model == 'lenet':
            self.model = self.LeNet5() 
        elif self.target_model == 'vgg':
            raise Exception(f'[ERROR]: vgg16 would be supported soon')
        else:
            raise Exception(f'[ERROR]: {self.target_model} is not supported in this version') 
        return


    def _set_experiment(self) -> None:
        # 1. get dataset
        self._gen_dataset()
        # 2. gen model.
        self._gen_model()
        # 3. train model
        
        return 




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=False)
    parser.add_argument('--model', help='target model(e.g. lenet, vgg)')
    parser.add_argument('--mode', help='method to generate adversarial attack(e.g. adapt, fgsm)')
    
    args = parser.parse_args()


if __name__ == "__main__":
    main()
