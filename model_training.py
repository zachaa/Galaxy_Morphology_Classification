import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D


r"""
To run
1. navigate to the main directory: `Galaxy_Morphology_Classification`
2. activate the correct virtual environment: `venv\scripts\activate`
3. run: `python model_training.py`

Partial output is saved to .log file while running.
The rest can be copy pasted if needed.
"""


RANDOM_STATE = 32  # DO NOT CHANGE, images will have to be resorted
SQL_DATABASE = "data/galaxy_data.sqlite"
# File containing all the training image data as an array of arrays (np.float32)
TRAIN_IMAGES_ARRAY_NPY = "data/training_images_array.npy"


# Output to log file and console
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(r"C:\Users\Zachary\Desktop\console_output.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def run_training(starting_epoch: int = 0, ending_epoch: int = 50):
    """
    Train the model by loading an already existing npy file containing all the training image data.

    This file is similar to `model_training.ipynb` with some changes to run
    just training in the console and not create any graphs or evaluating the
    model with testing data.

    Checkpoints are still created and a model (.keras) will be saved when training stops.

    :param starting_epoch: Epoch number to start on (same as previous ending_epoch), default to 0
    :param ending_epoch: Number of epochs to run for, defaults to 50
    """
    logging.info(f"Starting Epoch: {starting_epoch}, Ending Epoch: {ending_epoch}")

    # load from SQLite database
    connection = sqlite3.connect(SQL_DATABASE)
    df_import = pd.read_sql("SELECT * from galaxy_data", connection)
    connection.close()

    # Make sure there are no null values in data
    if df_import.isnull().any(axis=1).sum() != 0:
        logging.info("Exiting, there are nulls in the data.")
        return

    # keep only needed values
    stratify_data = df_import["class_reduced"].values
    x_image_id_names = df_import["asset_id"]
    y_output_data = df_import.drop(["objid", "sample", "asset_id", "dr7objid", "ra", "dec", "gz2_class", "class_reduced"], axis=1)
    logging.info(f"Full y_data: {y_output_data.shape}")

    # Split data into testing and training
    # X is asset names, not the actual images
    _, _, y_train, _ = train_test_split(x_image_id_names,
                                        y_output_data,
                                        random_state=RANDOM_STATE,
                                        stratify=stratify_data)
    y_train = y_train.astype("float32")

    # Load Training images
    if not Path(TRAIN_IMAGES_ARRAY_NPY).exists():
        logging.info(f".npy file not found at: {Path(TRAIN_IMAGES_ARRAY_NPY).absolute()}")
        logging.info("Exiting")
        return
    logging.info(" loading training images from npy file...")
    X_train_images = np.load(TRAIN_IMAGES_ARRAY_NPY)
    # Loading png images is not in this file
    logging.info(f"X_train_images Shape: {X_train_images.shape}")
    logging.info(f"X_train_images Size {X_train_images.nbytes} bytes")
    logging.info(f"y_train Shape: {y_train.shape}")

    # Callbacks for Early Stopping and Checkpoints
    # - https://www.tensorflow.org/tutorials/keras/save_and_load
    checkpoints = ModelCheckpoint("data/model/checkpoints/cp-{epoch:03d}.ckpt",
                                  monitor="loss", mode="min",
                                  save_weights_only=True,
                                  verbose=0)
    early_stopping = EarlyStopping(monitor="loss", patience=7)

    # Callbacks for Reducing Learning Rate
    def scheduler(epoch: int, lr: float) -> float:
        """Slightly reduce the learning rate every 10 epochs"""
        if epoch % 10 == 0 and epoch != 0:
            return lr * 0.8
        else:  # No change
            return lr
        
    lr_scheduler = LearningRateScheduler(scheduler, verbose=0)
    reduce_lr_plateau = ReduceLROnPlateau(monitor="loss",
                                          factor=0.1, min_lr=0.000_000_01,
                                          patience=5,
                                          verbose=0)

    # Logging (history from .fit() is only saved for the current run of the model)
    # CSVLogger can add new data to existing file to persist information
    csv_logger_start = CSVLogger("data/model/training_log.csv", separator=",", append=False)
    csv_logger_resume = CSVLogger("data/model/training_log.csv", separator=",", append=True)
    if starting_epoch == 0:
        callbacks_ = [checkpoints, early_stopping, lr_scheduler, reduce_lr_plateau, csv_logger_start]
    else:
        logging.info("Setting callbacks to use csv_logger_resume")
        callbacks_ = [checkpoints, early_stopping, lr_scheduler, reduce_lr_plateau, csv_logger_resume]

    def r2_score(y_true, y_pred):
        """Custom R Squared metric as R2Score() in tensorflow 2.13.0 causes type
        error and is not available in earlier versions"""
        SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
        SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
        return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

    if starting_epoch == 0:
        # ------------------------------------------------------------------------------
        # Create the Model
        # ------------------------------------------------------------------------------
        IMG_SIZE = X_train_images[0].shape[0]
        INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

        # Create a sequential model
        model = Sequential()

        # Add convolution layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2, seed=RANDOM_STATE))
        # Flatten the output from convolution layers
        model.add(Flatten())

        # Add dense (fully connected) layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2, seed=RANDOM_STATE))

        model.add(Dense(64, activation='relu'))

        # Add the output layer with 37 units (for 37 classes)
        model.add(Dense(37, activation='sigmoid'))

        # Compile the model
        # metric names: "root_mean_squared_error" and "r2_score"
        model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError(), r2_score])  # mse=mean_squared_error
    else:
        # Load already existing model to continue training
        model = load_model("data/model/GalaxyConfidenceModel.keras", custom_objects={"r2_score": r2_score})
        logging.info("Model loaded.")

    logging.info(f"{model.summary()}")

    # Verify start to training
    verify = input("Begin model training? (y/n)")
    if verify.lower() != "y":
        logging.info("Exiting without training.")
        # make sure files are removed from memory if not continuing
        del model
        del X_train_images
        del y_train
        return

    # ------------------------------------------------------------------------------
    # Train the model
    # ------------------------------------------------------------------------------
    logging.info(" Begin training.")
    start_training_time = time.time()

    history = model.fit(X_train_images,
                        y_train,
                        initial_epoch=starting_epoch,
                        epochs=ending_epoch,
                        callbacks=callbacks_,
                        batch_size=2_000,
                        validation_split=0.1)
    
    logging.info(" Training complete.")

    try:
        _hr, _remainder = divmod(time.time() - start_training_time, 3600)
        _min, _sec = divmod(_remainder, 60)
        logging.info(f"--- Time Taken: {int(_hr):02d}:{int(_min):02d}:{int(_sec):02d} ---")
    except Exception:  # just in case something goes wrong above
        logging.info(f"Start time: {start_training_time}")
        logging.info(f"End time: {time.time()}")

    # ------------------------------------------------------------------------------
    # Save the trained model
    # ------------------------------------------------------------------------------
    logging.info(" Saving Model...")
    model.save("data/model/GalaxyConfidenceModel.keras")
    logging.info("Saved Model!")

    logging.info("History:")
    logging.info(f"{history.history}")
    logging.info("")

    # make sure files are removed from memory
    del model
    del X_train_images
    del y_train


if __name__ == "__main__":
    logging.info("Starting program.")
    run_training(starting_epoch=0, ending_epoch=120)
    logging.info("Program complete.")
