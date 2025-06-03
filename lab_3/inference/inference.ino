
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

float zeroPoint = 0.0;
float scale = 0.0;

// Create a static memory buffer for TFLM, the size may need to

// be adjusted based on the model you are using

constexpr int tensorArenaSize = 16 * 1024;

byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

inline int8_t quantize_int8(float x, float scale, int zero_point)
{
  int q = static_cast<int>(std::lround(x / scale) + zero_point);
  if (q < -128)
    q = -128;
  if (q > 127)
    q = 127;
  return static_cast<int8_t>(q);
}

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

  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    Serial.println("Allocation failed!");
    while (1)
      ;
    return;
  }

  // Get pointers for the model's input and output tensors

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
  zeroPoint = tflInputTensor->params.zero_point;
  scale = tflInputTensor->params.scale;

  TfLiteIntArray *dims = tflInputTensor->dims;
  MicroPrintf("Model input rank = %d  (bytes=%d; type=%d)\n",
              dims->size, tflInputTensor->bytes, tflInputTensor->type);
  MicroPrintf("  dims = [");
  for (int i = 0; i < dims->size; i++)
  {
    MicroPrintf(" %d", dims->data[i]);
  }
  MicroPrintf(" ]\n");
}

void loop()
{

  float aX, aY, aZ;

  while (samplesRead == numSamples)
  {

    if (IMU.accelerationAvailable())
    {

      // read the acceleration data
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold

      if (aSum >= accelerationThreshold)
      {
        samplesRead = 0;
        break;
      }
    }
  }

  while (samplesRead < numSamples)
  {

    if (IMU.accelerationAvailable())
    {

      IMU.readAcceleration(aX, aY, aZ);

      // tflInputTensor->data.f[samplesRead * 3 + 0] = aY * 10;
      // tflInputTensor->data.f[samplesRead * 3 + 1] = aX * 10;
      // tflInputTensor->data.f[samplesRead * 3 + 2] = aZ * 10;
      tflInputTensor->data.int8[samplesRead * 3] = quantize_int8(aY * 10, scale, zeroPoint);
      tflInputTensor->data.int8[samplesRead * 3 + 1] = quantize_int8(aX * 10, scale, zeroPoint);
      tflInputTensor->data.int8[samplesRead * 3 + 2] = quantize_int8(aZ * 10, scale, zeroPoint);
      samplesRead++;

      if (samplesRead == numSamples)
      {

        // Run inferencing
        unsigned long start = millis();

        TfLiteStatus invokeStatus = tflInterpreter->Invoke();

        unsigned long duration = millis() - start;
        Serial.print("Inference time (ms): ");
        Serial.println(duration);

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
          Serial.println(tflOutputTensor->data.int8[i]);
        }

        Serial.println();
      }
    }
  }
}