
/*

  IMU Classifier

  This example uses the on-board IMU to start reading acceleration and gyroscope

  data from on-board IMU, once enough samples are read, it then uses a

  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.

  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally

        discouraged in Arduino examples, and in the future the TensorFlowLite library

        might change to make the sketch simpler.

  The circuit:

  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.

  Created by Don Coleman, Sandeep Mistry

  Modified by Dominic Pajak, Sandeep Mistry

  This example code is in the public domain.

*/

#include <Arduino_LSM9DS1.h>

#include <TensorFlowLite.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>

// #include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>

#include <tensorflow/lite/micro/micro_interpreter.h>

#include <tensorflow/lite/schema/schema_generated.h>

// #include <tensorflow/lite/version.h>

#include "gesture_model.h"
#include "raw_data.h"

const float accelerationThreshold = 0; // threshold of significant in G's

const int numSamples = 200;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)

// tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and

// only pull in the TFLM ops you need, if would like to reduce

// the compiled size of the sketch.

tflite::AllOpsResolver tflOpsResolver;

const tflite::Model *tflModel = nullptr;

tflite::MicroInterpreter *tflInterpreter = nullptr;

TfLiteTensor *tflInputTensor = nullptr;

TfLiteTensor *tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to

// be adjusted based on the model you are using

constexpr int tensorArenaSize = 128 * 1024;

byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

void setup()
{

  Serial.begin(9600);

  while (!Serial)
    ;

  Serial.println("=== MODEL PARAMS ===");

  // initialize the IMU

  if (!IMU.begin())
  {

    Serial.println("Failed to initialize IMU!");

    while (1)
      ;
  }

  // print out the samples rates of the IMUs

  Serial.print("Accelerometer sample rate = ");

  Serial.print(IMU.accelerationSampleRate());

  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array

  tflModel = tflite::GetModel(gesture_model);

  if (tflModel->version() != TFLITE_SCHEMA_VERSION)
  {

    Serial.println("Model schema mismatch!");

    while (1)
      ;
  }

  // Create an interpreter to run the model

  // tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  // Allocate memory for the model's input and output tensors

  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors

  tflInputTensor = tflInterpreter->input(0);

  tflOutputTensor = tflInterpreter->output(0);
}

void loop()
{

  float aX, aY, aZ;

  memcpy(tflInputTensor->data.f, idle_raw, 600 * sizeof(float));

  // Run inferencing
  Serial.println("First 10 inputs:");
  for (int i = 0; i < 10; i++)
  {
    Serial.println(tflInputTensor->data.f[i], 6);
  }
  Serial.println("Last 10 inputs:");
  for (int i = 589; i < 600; i++)
  {
    Serial.println(tflInputTensor->data.f[i], 6);
  }
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();

  if (invokeStatus != kTfLiteOk)
  {
    Serial.println("Invoke failed!");
    while (1)
      ;
    return;
  }

  // Loop through the output tensor values from the model

  for (int i = 0; i < available_classes_num; i++)
  {

    Serial.print(available_classes[i]);
    Serial.print(": ");
    Serial.println(tflOutputTensor->data.f[i], 6);
  }

  Serial.println();
  delay(5000);
}
