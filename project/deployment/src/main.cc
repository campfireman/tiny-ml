#include <Arduino.h>
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite_ESP32.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include <tensorflow/lite/schema/schema_generated.h>

#include "SoundAnalyzer.h"
#include "microphone.h"
#include "model.h"

using namespace SoundAnalyzer;
Analyzer<int16_t> Processor;

#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define GAIN 10.0
#define FRAME_SIZE 2048
#define HOP_SIZE 512
#define NUM_MFCC_COEFFS 13

int32_t rawBuf[RECORD_SAMPLES];
int16_t input[RECORD_SAMPLES];
int16_t frameBuf[FRAME_SIZE];
float mfccMatrix[32][NUM_MFCC_COEFFS];

tflite::AllOpsResolver tflOpsResolver;
const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;
tflite::ErrorReporter *error_reporter = nullptr;

float zeroPoint = 0.0;
float scale = 0.0;

constexpr int tensorArenaSize = 128 * 1024;

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
  Serial.begin(115200);
  microphoneInit(SAMPLE_RATE);

  // configure the analyzer
  AnalyzerConfig PConfig = Processor.defaultConfig();

  PConfig.samplefreq = SAMPLE_RATE;
  PConfig.gain = 0;
  PConfig.sensitivity = 0;
  PConfig.mfcccoeff = 13,
  Processor.setConfig(PConfig);

  tflModel = tflite::GetModel(model);

  if (tflModel->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("Model schema mismatch!");
    while (1)
      ;
  }

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, error_reporter);

  tflInterpreter->AllocateTensors();

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
  zeroPoint = tflInputTensor->params.zero_point;
  scale = tflInputTensor->params.scale;
}

void loop()
{
  uint32_t n = microphoneListen(rawBuf, RECORD_SAMPLES);
  for (int i = 0; i < n; ++i)
    frameBuf[i] = rawBuf[i] >> 16;
  int n_frames = 0;
  for (int off = 0; off + FRAME_SIZE <= RECORD_SAMPLES; off += HOP_SIZE)
  {
    for (int i = 0; i < FRAME_SIZE; ++i)
    {
      frameBuf[i] = rawBuf[off + i] >> 16;
    }
    Processor.doFft(frameBuf, true);
    float *coeffs = Processor.getMfcc(); // length = NUM_MFCC_COEFFS
    for (int j = 0; j < NUM_MFCC_COEFFS; ++j)
    {
      mfccMatrix[n_frames][j] = coeffs[j];
    }
    n_frames++;
    if (n_frames >= 32)
      break;
  }
  Serial.printf("MFCC shape: (%d, %d)\n", n_frames, NUM_MFCC_COEFFS);

  // write to model for inference
  int idx = 0;
  for (int i = 0; i < 28; ++i)
  {
    for (int j = 0; j < NUM_MFCC_COEFFS; ++j)
    {
      // tflInputTensor->data.int8[idx++] = quantize_int8(mfccMatrix[i][j], scale, zeroPoint);
      Serial.print("i:");
      Serial.print(i);
      Serial.print(" j:");
      Serial.print(j);
      Serial.println(quantize_int8(mfccMatrix[i][j], scale, zeroPoint));
    }
  }

  // TfLiteStatus invokeStatus = tflInterpreter->Invoke();

  // if (invokeStatus != kTfLiteOk)
  // {
  //   Serial.println("Invoke failed!");
  //   while (1)
  //     ;
  //   return;
  // }

  // for (int i = 0; i < available_classes_num; i++)
  // {

  //   Serial.print(available_classes[i]);
  //   Serial.print(": ");
  //   Serial.println(tflOutputTensor->data.int8[i]);
  // }

  // Serial.println();
}
