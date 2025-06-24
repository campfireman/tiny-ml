import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser("model_info")
parser.add_argument("path", help="The path to the model to inspect", type=str)
args = parser.parse_args()

# point this at the .tflite you think is int8!
model_path = args.path

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]

print(">>> input dtype:", inp["dtype"])
print(">>> input shape:", inp["shape"])
print(">>> quant params:", inp["quantization"])

