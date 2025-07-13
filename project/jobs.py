import os
import re
from abc import ABC
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import List

import librosa
import matplotlib.pyplot as plt
import models
import numpy as np
import pandas as pd
import seaborn as sns
import speechpy
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from tensorflow import keras

STORAGE = "storage"
DATA_DIR = "data"
TRAIN_DATASET_RATIO = 0.2


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels: List[str]


@dataclass
class TrainingResult:
    model: object
    dataset: Dataset


@dataclass
class OptimizationResult:
    optimized_model: object
    dataset: Dataset


class Job(ABC):
    def get_run_dir(self):
        return Path(os.path.join(STORAGE, self.get_job_name(), self.run_id))

    @staticmethod
    def get_job_name():
        raise NotImplementedError("Job needs a name!")


class FeatureExtraction(Job):

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
            return self.load()
        else:
            print(f"Creating features for {self.run_id}.")
            run_dir.mkdir(parents=True)
            dataset = self.create_features()
            self.store(dataset)
            return dataset

    def visualize_mfcc(self, sr, mfcc, label):
        plt.figure(figsize=(10, 4))
        print(mfcc)
        librosa.display.specshow(mfcc, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title(f"MFCC - {label}")
        plt.tight_layout()
        plt.show()

    def perturb(self, y: np.ndarray, sr: int) -> np.ndarray:
        choice = np.random.choice(["noise", "pitch"], p=[
                                  0.5, 0.5])
        if choice == "noise":
            noise_lvl = 0.005 * np.random.randn(len(y))
            return y + noise_lvl
        elif choice == "pitch":
            n_steps = np.random.uniform(-2, 2)
            return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        else:
            return y

    def pad(self, yy, target_len=16000):
        if len(yy) >= target_len:
            return yy[:target_len]
        print("padding data...")
        padding_len = target_len - len(yy)
        noise = np.concatenate([yy[:200], yy[-200:]])
        pad = np.random.normal(np.mean(noise), np.std(
            noise) + 1e-6, size=padding_len)
        return np.concatenate([yy, pad])

    def create_features(self):
        data_dir = Path(DATA_DIR)
        labels = []
        samples = []

        for file in data_dir.iterdir():
            if not file.is_file():
                continue
            m = re.search(
                r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-(\w+)", file.name)
            if not m:
                continue
            extracted_label = m.group(1)
            # if extracted_label in {"zenmode"}:
            #     continue
            if extracted_label not in labels:
                labels.append(extracted_label)
            label = labels.index(extracted_label)
            yy, sr = librosa.load(file, sr=None)
            yy = self.pad(yy)

            # original
            samples.append((yy, sr, label))
            # augmented
            if extracted_label not in {"unknown", "idle"}:
                y_aug = self.perturb(yy, sr)
                samples.append((y_aug, sr, label))
                y_aug = self.perturb(yy, sr)
                samples.append((y_aug, sr, label))

        x, y = [], []
        for yy, sr, label in samples:
            frame_length = 2048 / sr  # 0.128 s
            frame_stride = 512 / sr  # 0.032 s

            mfcc = speechpy.feature.mfcc(
                yy,
                sampling_frequency=sr,
                frame_length=frame_length,
                frame_stride=frame_stride,
                num_cepstral=13,
                num_filters=40,
                fft_length=2048,
                low_frequency=0,
                high_frequency=None,
                dc_elimination=True
            ).flatten().reshape((-1, 1))

            x.append(mfcc)
            y.append(label)

        expected_shape = x[0].shape
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
        y_train = keras.utils.to_categorical(y_train, len(labels))
        y_test = keras.utils.to_categorical(y_test, len(labels))

        return Dataset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            labels=labels,
        )

    def load(self):
        run_dir = Path(STORAGE) / self.get_job_name() / self.run_id
        archive = run_dir / "features.npz"
        if not archive.exists():
            raise FileNotFoundError(f"No feature archive found at {archive}")
        with np.load(archive) as data:
            return Dataset(
                x_train=data["x_train"],
                y_train=data["y_train"],
                x_test=data["x_test"],
                y_test=data["y_test"],
                labels=data["labels"],
            )

    def store(self, dataset: Dataset):
        run_dir = Path(STORAGE) / self.get_job_name() / self.run_id
        archive = run_dir / "features.npz"
        # savez_compressed gives smaller files at the cost of a bit more CPU time
        np.savez_compressed(
            archive,
            x_train=dataset.x_train,
            y_train=dataset.y_train,
            x_test=dataset.x_test,
            y_test=dataset.y_test,
            labels=dataset.labels,
        )


class Training(Job):

    def __init__(self, run_id):
        self.run_id = run_id

    @staticmethod
    def get_job_name():
        return "train"

    def __repr__(self):
        return "Model Training"

    def train(self, dataset):
        model = self.build_model(dataset)
        model.summary()
        early_stopping_cb = EarlyStopping(
            # monitor="val_loss",
            monitor="loss",
            patience=40,
            # min_delta=0.01,
            mode='min',
            restore_best_weights=True
        )
        callbacks = [early_stopping_cb, ]

        num_epochs = 200
        history = model.fit(
            dataset.x_train, dataset.y_train, batch_size=128,
            validation_data=(dataset.x_test, dataset.y_test),
            epochs=num_epochs, callbacks=callbacks,
        )
        self.plot_training_history(history, 1)
        return model

    def execute(self, dataset):
        run_dir = self.get_run_dir()

        if run_dir.exists():
            print(f"Loading model {self.run_id}.")
            model = self.load_model()
        else:
            print(f"Training model {self.run_id}.")
            run_dir.mkdir(parents=True)
            model = self.train(dataset)
            self.store_model(model)

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

    def build_model(self, dataset):
        # model = models.get_residual_model((351, 1), len(dataset.labels))
        # model = models.get_convolutional_model((351, 1), len(dataset.labels))
        model = models.get_ds_cnn_model((351, 1), len(dataset.labels))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        return model

    def load_model(self):
        model = keras.models.load_model(self.build_model_path())
        model.summary()
        return model

    def store_model(self, model):
        model.save(self.build_model_path())

    def build_model_path(self):
        return os.path.join(
            self.get_run_dir(), f"{self.run_id}.keras")


class Evaluation(Job):

    def __init__(self, run_id):
        self.run_id = run_id

    @staticmethod
    def get_job_name():
        return "evaluation"

    def __repr__(self):
        return "Model Evaluation"

    def execute(self, training_result: TrainingResult):
        model = training_result.model
        ds = training_result.dataset
        score_model = model.evaluate(ds.x_test, ds.y_test)  # , verbose=0)
        print("Test loss:", score_model[0])
        print("Test accuracy:", score_model[1])

        cm = confusion_matrix(np.argmax(ds.y_test, axis=1),
                              np.argmax(model.predict(ds.x_test), axis=1))
        # print(cm)

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm = pd.DataFrame(cm, index=ds.labels,
                          columns=ds.labels)

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

        return training_result


class Optimization(Job):

    def __init__(self, run_id):
        self.run_id = run_id

    @staticmethod
    def get_job_name():
        return "optimization"

    def __repr__(self):
        return "Model Optimization"

    def execute(self, training_result):
        run_dir = self.get_run_dir()

        if run_dir.exists():
            print(f"Loading optimized model {self.run_id}.")
            optimized_model = self.load()
        else:
            print(f"Optimizing model {self.run_id}.")
            optimized_model = self.optimize(
                training_result.model,
                training_result.dataset,
            )
            run_dir.mkdir(parents=True)
            self.store(optimized_model)

        self.evaluate(
            training_result.dataset,
            training_result.model,
            optimized_model,
        )

        return OptimizationResult(
            dataset=training_result.dataset,
            optimized_model=optimized_model
        )

    def representative_dataset_gen(self, dataset):
        def generator():
            for i in range(len(dataset.x_train)):
                yield [np.expand_dims(dataset.x_train[i].astype(np.float32), axis=0)]
        return generator

    def optimize(self, model, dataset):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        converter.representative_dataset = self.representative_dataset_gen(
            dataset)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        return converter.convert()

    def build_model_path(self):
        return os.path.join(
            self.get_run_dir(), f"{self.run_id}.tflite")

    def load(self):
        with open(self.build_model_path(), 'rb') as f:
            return f.read()

    def store(self, optimized_model):
        with open(self.build_model_path(), 'wb') as f:
            f.write(optimized_model)

    def convert_input(self, x, dtype, input_details):
        if dtype == np.int8:
            scale, zp = input_details[0]["quantization"]
            q = np.round(x/scale + zp) \
                .clip(-128, 127) \
                .astype(np.int8)
            return q.reshape((1, 351, 1))
        else:
            return np.expand_dims(x.astype(dtype), axis=0)

    def evaluate(self, dataset, model, optimized_model):
        self.evaluate_model(model, dataset)
        self.evaluate_model(optimized_model, dataset)

    def evaluate_model(self, model, dataset):
        if isinstance(model, keras.Model):
            predictions = model.predict(dataset.x_test, batch_size=1)
            pred_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(dataset.y_test, axis=1)
            accuracy = np.mean(pred_labels == true_labels)
            print(f"Model accuracy: {accuracy:.4f}")
            print("Estimated RAM usage: N/A for Keras model")
        else:
            interpreter = tf.lite.Interpreter(model_content=model)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            dtype = input_details[0]["dtype"]
            output_details = interpreter.get_output_details()

            correct = 0
            total = len(dataset.x_test)
            for i in range(total):
                input_data = self.convert_input(
                    dataset.x_test[i], dtype, input_details)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                pred_label = np.argmax(output)
                true_label = np.argmax(dataset.y_test[i])
                if pred_label == true_label:
                    correct += 1

            accuracy = correct / total
            print(f"Model accuracy: {accuracy:.4f}")

            tensor_details = interpreter.get_tensor_details()
            total_tensor_bytes = 0
            for tensor in tensor_details:
                shape = tensor['shape']
                dtype = np.dtype(tensor['dtype'])
                size = np.prod(shape) * dtype.itemsize
                total_tensor_bytes += size

            print(
                f"Estimated RAM usage (tensor buffers only): {total_tensor_bytes / 1024:.2f} KB")


class Deployment(Job):
    def __init__(self, run_id):
        self.run_id = run_id

    @staticmethod
    def get_job_name():
        return "deployment"

    def __repr__(self):
        return "Model Deployment"

    def bytes_to_c_array(self, data: bytes, per_line: int = 12) -> str:
        hex_bytes = [f"0x{b:02x}" for b in data]
        lines = []
        for i in range(0, len(hex_bytes), per_line):
            lines.append(", ".join(hex_bytes[i: i + per_line]))
        return ",\n".join("    " + line for line in lines)

    def execute(self, optimization_result: OptimizationResult):
        run_dir = self.get_run_dir()
        labels = optimization_result.dataset.labels

        if not run_dir.exists():
            run_dir.mkdir(parents=True)

        data = optimization_result.optimized_model
        library_name = "model"
        arr_literal = self.bytes_to_c_array(data)

        max_len = max(len(lbl) for lbl in labels) + 1

        labels_entries = ",\n".join(f'    "{lbl}"' for lbl in labels)
        labels_cc = (
            f"const char available_classes[][ {max_len} ] = {{\n"
            f"{labels_entries}\n"
            f"}};\n"
        )

        model_cc = (
            f"alignas(16) const unsigned char {library_name}[] = {{\n"
            f"{arr_literal}\n"
            f"}};\n"
            f"const int {library_name}_len = {len(data)};\n"
        )

        export = Path(run_dir)
        cc_path = export / f"{library_name}.cc"
        with cc_path.open("w") as f:
            f.write(f'#include "{library_name}.h"\n\n')
            f.write(labels_cc + "\n")
            f.write(
                f"const int available_classes_num = {len(labels)};\n\n")
            f.write(model_cc)

        header = []
        header.append("#ifndef TENSORFLOW_LITE_MODEL_H_")
        header.append("#define TENSORFLOW_LITE_MODEL_H_\n")
        header.append("// Classes that can be detected by the neural network")
        header.append(f"extern const char available_classes[][ {max_len} ];")
        header.append("extern const int available_classes_num;\n")
        header.append("// Pre-trained neural network")
        header.append(
            f"alignas(16) extern const unsigned char {library_name}[];")
        header.append(f"extern const int {library_name}_len;\n")
        header.append("#endif /* TENSORFLOW_LITE_MODEL_H_ */\n")

        h_path = export / f"{library_name}.h"
        with h_path.open("w") as f:
            f.write("\n".join(header))

        print(f"Wrote {cc_path} and {h_path}")
        return
