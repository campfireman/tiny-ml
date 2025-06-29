#include <Arduino.h>
#include <Arduino_LSM9DS1.h>

#include <edge-impulse-sdk/tensorflow/lite/micro/all_ops_resolver.h>
#include <edge-impulse-sdk/tensorflow/lite/micro/micro_interpreter.h>
#include <edge-impulse-sdk/tensorflow/lite/schema/schema_generated.h>

#include "edge-impulse-sdk/dsp/speechpy/feature.hpp"
#include "edge-impulse-sdk/dsp/ei_vector.h"
#include "edge-impulse-sdk/dsp/returntypes.hpp"
#include "edge-impulse-sdk/dsp/ei_vector.h"

#include "microphone.h"
#include "model.h"
#include "raw_data.h"

#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define FRAME_SIZE 2048
#define HOP_SIZE 512
#define NUM_MFCC_COEFFS 13

int32_t rawBuf[RECORD_SAMPLES];
float mfccMatrix[32][NUM_MFCC_COEFFS];

tflite::AllOpsResolver tflOpsResolver;
const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;

float zeroPoint = 0.0;
float scale = 0.0;

constexpr int tensorArenaSize = 16 * 1024;

byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

static int ei_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
  // rawBuf holds your int32_t samples >>16 → int16_t
  for (size_t i = 0; i < length; i++)
  {
    // convert each sample to float in [–1, 1]
    int16_t s = rawBuf[offset + i] >> 16;
    out_ptr[i] = float(s) / 32768.0f;
  }
  return ei::EIDSP_OK;
}

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
  Serial.begin(115200);
  microphoneInit(SAMPLE_RATE);

  tflModel = tflite::GetModel(model);

  if (tflModel->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("Model schema mismatch!");
    while (1)
      ;
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  if (tflInterpreter->AllocateTensors(true) != kTfLiteOk)
  {
    Serial.println("Allocation failed!");
    while (true)
      ;
  }

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
  zeroPoint = tflInputTensor->params.zero_point;
  scale = tflInputTensor->params.scale;
}

void printHeapInfo()
{
  Serial.print("Heap: ");
  Serial.println(ESP.getHeapSize());
  Serial.print("Free Heap: ");
  Serial.println(ESP.getFreeHeap());
  Serial.print("PSRAM: ");
  Serial.println(ESP.getPsramSize());
  Serial.print("Free PSRAM: ");
  Serial.println(ESP.getFreePsram());
}

void loop()
{

  Serial.println("rec");
  uint32_t n = microphoneListen(rawBuf, RECORD_SAMPLES);
  Serial.println("st");

  unsigned long start = millis();
  // 1. Create the signal struct:
  ei::signal_t signal;
  signal.total_length = RECORD_SAMPLES;
  signal.get_data = ei_signal_get_data;

#define MAX_FRAMES (1 + (RECORD_SAMPLES - FRAME_SIZE) / HOP_SIZE)
  static float mfccMatrix[MAX_FRAMES][NUM_MFCC_COEFFS];

  int num_frames = MAX_FRAMES;

  // Construct in one shot:
  ei::matrix_t mfcc_out(27,
                        NUM_MFCC_COEFFS,
                        &mfccMatrix[0][0]);
  unsigned long signal_duration = millis() - start;

  int32_t status = ei::speechpy::feature::mfcc(
      &mfcc_out,
      &signal,
      SAMPLE_RATE,                      // 16000
      float(FRAME_SIZE) / SAMPLE_RATE,  // e.g. 2048/16000 = 0.128 s
      float(HOP_SIZE) / SAMPLE_RATE,    // e.g. 512/16000  = 0.032 s
      /*num_cepstral*/ NUM_MFCC_COEFFS, // e.g. 13
      /*num_filters*/ 40,
      /*fft_length*/ FRAME_SIZE, // must be ≥ frame_length in samples
      /*low_freq*/ 0,
      /*high_freq*/ SAMPLE_RATE / 2,
      /*append_energy*/ true,
      /*lifter*/ 1);
  if (status != ei::EIDSP_OK)
  {
    Serial.print("MFCC error, code = ");
    Serial.println(status);
    // Optionally decode into human-readable form:
    switch (status)
    {
    case ei::EIDSP_OK:
      Serial.println("OK");
      break;
    case ei::EIDSP_OUT_OF_MEM:
      Serial.println("OUT_OF_MEM");
      break;
    case ei::EIDSP_OUT_OF_BOUNDS:
      Serial.println("MATRIX_OUT_OF_BOUNDS");
      break;
    case ei::EIDSP_PARAMETER_INVALID:
      Serial.println("NOT_VALID_PARAM");
      break;
    // … add other EIDSP_* codes from dsp/returntypes.hpp …
    default:
      Serial.println("UNKNOWN_ERROR");
      break;
    }
  }
  unsigned long preprocessing_duration = millis() - start;

  // write to model for inference
  int idx = 0;
  for (int i = 0; i < 27; i++)
  {
    for (int j = 0; j < NUM_MFCC_COEFFS; j++)
    {
      float val_f = mfccMatrix[i][j];
      int8_t val_q = quantize_int8(val_f, scale, zeroPoint);
      tflInputTensor->data.int8[idx++] = val_q;
      // Serial.print(val_f, 6);
      // Serial.print(",");
    }
    // Serial.println();
  }

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();

  if (invokeStatus != kTfLiteOk)
  {
    Serial.println("Invoke failed!");
    while (1)
      ;
    return;
  }

  unsigned long full_duration = millis() - start;

  int8_t max = INT8_MIN;
  const char *label;
  for (int i = 0; i < available_classes_num; i++)
  {

    if (tflOutputTensor->data.int8[i] > max)
    {
      max = tflOutputTensor->data.int8[i];
      label = available_classes[i];
    }
    Serial.print(available_classes[i]);
    Serial.print(": ");
    Serial.println(tflOutputTensor->data.int8[i]);
  }

  Serial.println("---");
  Serial.print("Classification: ");
  Serial.println(label);
  Serial.println();

  Serial.print("Signal took ");
  Serial.print(signal_duration);
  Serial.println(" milliseconds");
  Serial.print("Preprocessing took ");
  Serial.print(preprocessing_duration);
  Serial.println(" milliseconds");
  Serial.print("Full took ");
  Serial.print(full_duration);
  Serial.println(" milliseconds");

  Serial.println();
}
