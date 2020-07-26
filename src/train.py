from models import get_vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow as tf
import argparse
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train_model(model, train_data, val_data, lr=0.001, batch_size= 16, froze_layer_index=20,
                optimizer=tf.keras.optimizers.Adam(), epoch=10):

    """ Parsing arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--froze_layer_index", type=int, default=20,
                        help="Froze layer index")
    parser.add_argument("--epoch", type=int, default=10,
                        help="Epoch")
    args = parser.parse_args()

    """Creating Data Generator"""

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        brightness_range=[0.6, 1.0],
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    train_generator = train_datagen.flow_from_dataframe(
        train_data,
        x_col='filename',
        y_col='category_names',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=args.batch_size
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_data,
        x_col='filename',
        y_col='category_names',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=args.batch_size
    )
    """Callbacks"""

    earlystop = EarlyStopping(patience=10)
    lr_scheduler = ReduceLROnPlateau(monitor='val_acc',
                                     patience=2,
                                     verbose=1,
                                     factor=0.1,
                                     min_lr=args.lr)

    callbacks = [earlystop, lr_scheduler]

    """Model compile and Train"""

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    history = model.fit_generator(
        train_generator,
        epochs=args.epoch,
        validation_data=val_generator,
        validation_steps=10,
        steps_per_epoch=10,
        callbacks=callbacks
    )
    model.save_weights('../weights/my_weights')

    return history


if __name__ == "__main__":

    train_data = pd.read_csv("../data/images/animal_train_data.csv")
    val_data = pd.read_csv("../data/images/animal_val_data.csv")
    model = get_vgg16()
    train_model(model,train_data,val_data)



