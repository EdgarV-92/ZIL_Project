from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as Layers
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras.models import Model


def get_vgg16(input_shape=(224, 224, 3), froze_layer_index=20):
    conv_base = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

    for layer in conv_base.layers[:froze_layer_index]:
        layer.trainable = False

    for layer in conv_base.layers[froze_layer_index:]:
        layer.trainable = True

    last_output = conv_base.layers[-1].output

    x = GlobalMaxPooling2D()(last_output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(conv_base.input, x)

    return model

# model = get_vgg16()
# print(model.summary())
