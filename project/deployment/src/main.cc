#include <Arduino.h>
#include <Arduino_LSM9DS1.h>

#include "audio_buffer.hpp"
#include "inference.hpp"
#include "microphone.h"
#include "model.h"
#include "mqtt.h"
#include "preprocessing.hpp"
#include "wifi.hpp"

#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define SEGMENT_NUMBER 4
#define SEGMENT_LENGTH (RECORD_SAMPLES / SEGMENT_NUMBER)
#define NUM_MFCC_COEFFS 13
#define CORE_0 0
#define CORE_1 1

#define DEBUG_MQTT 1
#define DEBUG_PERFORMANCE 0

static QueueHandle_t audioQueue;
static QueueHandle_t messageQueue;

void listen(void *pvParameters);
void process(void *pvParameters);
void message(void *pvParameters);

int32_t rawBuf[RECORD_SAMPLES];
float mfccMatrix[32][NUM_MFCC_COEFFS];

static AudioBuffer *audio_buf;

void setup()
{
  Serial.begin(115200);

  microphoneInit(SAMPLE_RATE);
  inferenceInit();
  wifiInit();
  mqttInit();
  audio_buf = new AudioBuffer(SEGMENT_LENGTH * (SEGMENT_NUMBER + 1), RECORD_SAMPLES);

  messageQueue = xQueueCreate(10, sizeof(u_int8_t));
  audioQueue = xQueueCreate(10, sizeof(int32_t *));

  if (messageQueue == NULL)
  {
    Serial.println("Failed to create command queue.");
    for (;;)
      ;
  }

  xTaskCreatePinnedToCore(
      listen,
      "Audio listening",
      2048,
      NULL,
      1,
      NULL,
      CORE_0);

  xTaskCreatePinnedToCore(
      process,
      "Data processing",
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

void listen(void *pvParameters)
{
  int initialSegmentCount = 0;
  for (;;)
  {
    uint32_t n = microphoneListen(audio_buf->startFillSegment(), SEGMENT_LENGTH);
    audio_buf->stopFillSegment(n);
    if (initialSegmentCount < SEGMENT_NUMBER)
    {
      initialSegmentCount++;
      continue;
    }
    int32_t *windowPtr = audio_buf->getLatestWindow();
    xQueueSend(audioQueue, &windowPtr, portMAX_DELAY);
  }
}

void process(void *pvParameters)
{

  int32_t *window;
  for (;;)
  {
    if (xQueueReceive(audioQueue, &window, portMAX_DELAY) != pdTRUE)
    {
      continue;
    }

    unsigned long start = millis();

    mfcc(window, mfccMatrix);

    unsigned long preprocessing_duration = millis() - start;

    uint8_t label_pos = infer(mfccMatrix);
    const char *label = available_classes[label_pos];

    unsigned long classification_duration = millis() - start;

    xQueueSend(messageQueue, (void *)&label_pos, portMAX_DELAY);

    unsigned long full_duration = millis() - start;

#if DEBUG_PERFORMANCE == 1
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
#endif
  }
}

void message(void *pvParameters)
{
  uint8_t pos;
  for (;;)
  {
    if (xQueueReceive(messageQueue, &pos, portMAX_DELAY) != pdTRUE)
    {
      continue;
    }
#if DEBUG_MQTT == 0
    mqttSend(available_classes[pos]);
#else
    Serial.print("Sent MQTT command: ");
    Serial.println(available_classes[pos]);
#endif
  }
}
