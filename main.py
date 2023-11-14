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
        self.mode = mode


    def _gen_dataset(self) -> Dict[str, Any]:
        pass


    def _set_experiment(self) -> None:
        # 1. get dataset
        # 2. gen model.
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
