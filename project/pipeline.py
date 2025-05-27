import argparse
from math import floor
from dataclasses import dataclass
import librosa
import re
from datetime import datetime
from pathlib import Path
import os
import random
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, ReLU, Add, Input, SeparableConv1D
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


ADJECTIVES = [
    "cheerful", "playful", "brilliant", "friendly", "eager", "happy",
    "jolly", "gentle", "kind", "lively", "witty", "bubbly", "sunny", "sparkling"
]
NOUNS = [
    "ada", "turing", "hopper", "compiler", "lambda", "socket", "protocol",
    "array", "kernel", "stack", "recursion", "segment", "pointer", "scheduler",
    "quine", "buffer", "thread", "bytecode", "pipeline", "neuron"
]
STORAGE = "storage"
DATA_DIR = "data"
TRAIN_DATASET_RATIO = 0.2
LABELS = ["idle", "chillaxo"]
NUM_CLASSES = len(LABELS)


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass
class TrainingResult:
    model: object
    dataset: Dataset


def random_word_pair():
    return f"{random.choice(ADJECTIVES)}_{random.choice(NOUNS)}"


class FeatureExtraction:

    def __init__(self, run_id):
        self.run_id = run_id

    @staticmethod
    def get_job_name():
        return "features"

    def __repr__(self):
        return "Feature Extraction"

    def execute(self, _):
        run_dir = Path(os.path.join(STORAGE, self.get_job_name(), self.run_id))

        if run_dir.exists():
            print(f"Loading features for {self.run_id}.")
            return self.load_features()
        else:
            print(f"Creating features for {self.run_id}.")
            run_dir.mkdir(parents=True)
            dataset = self.create_features()
            self.store(dataset)
            return dataset

    def create_features(self):
        data_dir = Path(DATA_DIR)
        files = []
        for file in data_dir.iterdir():
            if not file.is_file():
                continue

            match = re.search(
                r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-(\w+)", file.name)
            if not match:
                continue
            label = match.group(1)
            files.append((file, label))

        x = []
        y = []

        for file, label in files:
            yy, sr = librosa.load(file, sr=None)
            mfcc = librosa.feature.mfcc(y=yy, sr=sr, n_mfcc=13)
            mfcc = mfcc.T
            x.append(mfcc)
            y.append(LABELS.index(label))

        expected_shape = x[0].shape
        print(expected_shape)
        if not all(sample.shape == expected_shape for sample in x):
            raise ValueError("Inconsistent MFCC shapes in dataset")

        x = np.stack(x).astype(np.float32)
        y = np.array(y)

        n_data = len(x)
        train_end = floor(TRAIN_DATASET_RATIO * n_data)
        p = np.random.permutation(n_data)
        x_train = x[p[:train_end]]
        y_train = y[p[:train_end]]
        x_test = x[p[train_end:]]
        y_test = y[p[train_end:]]
        y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

        return Dataset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    def load_features(self):
        pass

    def store(self, dataset):
        pass


class ModelTraining:

    def __init__(self, run_id):
        self.run_id = run_id

    @staticmethod
    def get_job_name():
        return "train"

    def __repr__(self):
        return "Model Training"

    def execute(self, dataset):
        model = self.build_model()
        model.summary()
        early_stopping_cb = EarlyStopping(
            monitor="val_loss",
            patience=10,
            # min_delta=0.01,
            mode='min',
            restore_best_weights=True
        )
        callbacks = [early_stopping_cb, ]

        num_epochs = 50
        history = model.fit(dataset.x_train, dataset.y_train, batch_size=128,
                            epochs=num_epochs, validation_split=0.1, callbacks=callbacks)
        self.plot_training_history(history, 1)
        return TrainingResult(
            model=model,
            dataset=dataset
        )

    def plot_training_history(self, history, model_name):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f'Model {model_name}')
        fig.set_figwidth(15)

        ax1.plot(
            range(1, len(history.history['accuracy'])+1), history.history['accuracy'])
        ax1.plot(range(
            1, len(history.history['val_accuracy'])+1), history.history['val_accuracy'])
        ax1.set_title('Model accuracy')
        ax1.set(xlabel='epoch', ylabel='accuracy')
        ax1.legend(['training', 'validation'], loc='best')

        ax2.plot(
            range(1, len(history.history['loss'])+1), history.history['loss'])
        ax2.plot(
            range(1, len(history.history['val_loss'])+1), history.history['val_loss'])
        ax2.set_title('Model loss')
        ax2.set(xlabel='epoch', ylabel='loss')
        ax2.legend(['training', 'validation'], loc='best')
        plt.show()

    def build_model(summary=True):
        # inp = Input(shape=(200,3))

        # # Initial separable conv
        # x = SeparableConv1D(8, 5, activation='relu', padding='same')(inp)
        # x = BatchNormalization()(x)
        # x = MaxPooling1D(2)(x)

        # # Bottleneck + dilated conv
        # x = bottleneck_block(x, filters=16, dilation_rate=2)
        # x = BatchNormalization()(x)
        # x = MaxPooling1D(2)(x)

        # # Residual refinement
        # x = residual_block(x, filters=16)
        # x = MaxPooling1D(2)(x)

        # # Classifier head
        # x = Flatten()(x)
        # x = Dense(16, activation='relu',
        #         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        # x = Dropout(0.3)(x)
        # out = Dense(num_classes, activation='softmax', name='y_pred')(x)

        # model = Model(inputs=inp, outputs=out)

        model = Sequential([
            Conv1D(16, kernel_size=5, activation='relu', input_shape=(32, 13)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            Conv1D(32, kernel_size=5, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            Flatten(),

            Dense(32, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dropout(0.3),

            Dense(NUM_CLASSES, activation='softmax', name='y_pred')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


class ModelEvaluation:

    def __init__(self, run_id):
        self.run_id = run_id

    @staticmethod
    def get_job_name():
        return "features"

    def __repr__(self):
        return "Model Evaluation"

    def execute(self, training_result):
        model = training_result.model
        ds = training_result.dataset
        score_model = model.evaluate(ds.x_test, ds.y_test)  # , verbose=0)
        print("Test loss:", score_model[0])
        print("Test accuracy:", score_model[1])

        cm = confusion_matrix(np.argmax(ds.y_test, axis=1),
                              np.argmax(model.predict(ds.x_test), axis=1))
        # print(cm)

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm = pd.DataFrame(cm, index=LABELS,
                          columns=LABELS)

        plt.figure(figsize=(4, 4))
        ax = sns.heatmap(cm*100,
                         annot=True,
                         fmt='.1f',
                         cmap="Blues",
                         cbar=False,
                         )
        ax.set_ylabel("True Class", fontdict={'fontweight': 'bold'})
        ax.set_xlabel("Predicted Class", fontdict={'fontweight': 'bold'})

        plt.show()


PIPELINE = [
    FeatureExtraction,
    ModelTraining,
    ModelEvaluation
    # quantize/optimize
    # create embedded
]


def find_start(job_name, pipeline):
    for i, job in enumerate(pipeline):
        if job.get_job_name() == job_name:
            return i
    return None


def main():
    parser = argparse.ArgumentParser("Training Pipeline")
    parser.add_argument("--job_name", type=str, required=False)
    parser.add_argument("--run_id", type=str, required=False)
    args = parser.parse_args()

    job_name = args.job_name.lower(
    ) if args.job_name else PIPELINE[0].get_job_name()
    run_id = args.run_id

    if job_name != PIPELINE[0].get_job_name() and not run_id:
        raise ValueError(
            f"When specifying job {job_name} run_id must be given!")

    if not run_id:
        run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{random_word_pair()}"

    start = find_start(job_name, PIPELINE)

    if start == None:
        raise ValueError(f"No pipeline job with name \"{job_name}\" found!")

    input = None
    for job in PIPELINE[start:]:
        job = job(run_id)
        print(f"[ ] Running job {job}...")
        input = job.execute(input)
        print(f"[x] Job {job} done!")


if __name__ == "__main__":
    main()
