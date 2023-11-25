from typing import *


# LeNet-4 모델 구성
def LeNet5():
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
    x = Convolution2D(4, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    
    # Block 2.
    x = Convolution2D(16, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    
    # Fully connected.
    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    # x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax', name='redictions')(x)
    
    return Model(input_tensor, x)


def main():
    model = LeNet5()
    model.summary()


if __name__ == "__main__":
    main()
