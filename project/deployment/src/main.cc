#include <Arduino.h>
#include <Arduino_LSM9DS1.h>

#include "edge-impulse-sdk/dsp/speechpy/feature.hpp"
#include "edge-impulse-sdk/dsp/ei_vector.h"
#include "edge-impulse-sdk/dsp/returntypes.hpp"
#include "edge-impulse-sdk/dsp/ei_vector.h"

#include "microphone.h"
#include "inference.hpp"
#include "model.h"
#include "wifi.hpp"
#include "mqtt.h"

#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define FRAME_SIZE 2048
#define HOP_SIZE 512
#define NUM_MFCC_COEFFS 13
#define CORE_0 0
#define CORE_1 1

static QueueHandle_t commandQueue;

void inference(void *pvParameters);
void message(void *pvParameters);

int32_t rawBuf[RECORD_SAMPLES];
float mfccMatrix[32][NUM_MFCC_COEFFS];

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

void setup()
{
  Serial.begin(115200);
  microphoneInit(SAMPLE_RATE);

  inferenceInit();

  wifiInit();
  mqttInit();

  commandQueue = xQueueCreate(10, sizeof(u_int8_t)); // 10 entries

  if (commandQueue == NULL)
  {
    Serial.println("Failed to create command queue.");
    for (;;)
      ;
  }

  xTaskCreatePinnedToCore(
      inference,
      "Task Inferennce",
      2048,
      NULL,
      1,
      NULL,
      CORE_0);

  xTaskCreatePinnedToCore(
      message,
      "Task message to MQTT",
      2048,
      NULL,
      1,
      NULL,
      CORE_1);
}

void loop()
{
}

void inference(void *pvParameters)
{

  for (;;)
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

    uint8_t label_pos = infer(mfccMatrix);
    const char *label = available_classes[label_pos];

    unsigned long classification_duration = millis() - start;

    xQueueSend(commandQueue, (void *)&label_pos, portMAX_DELAY);

    unsigned long full_duration = millis() - start;

    Serial.println("---");
    Serial.print("Classification: ");
    Serial.println(label);
    Serial.println();

    Serial.print("Preprocessing took ");
    Serial.print(preprocessing_duration);
    Serial.println(" milliseconds");

    Serial.print("Classification took ");
    Serial.print(classification_duration);
    Serial.println(" milliseconds");

    Serial.print("Full took ");
    Serial.print(full_duration);
    Serial.println(" milliseconds");

    Serial.println();
  }
}

void message(void *pvParameters)
{
  uint8_t pos;
  for (;;)
  {
    if (xQueueReceive(commandQueue, &pos, portMAX_DELAY) == pdTRUE)
    {
      mqttSend(available_classes[pos]);
      Serial.println("Send command.");
    }
  }
}
