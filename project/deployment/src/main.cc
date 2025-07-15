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

#define DEBUG_MQTT 0
#define DEBUG_PERFORMANCE 0
#define DEBUG_AUDIO_PERFORMANCE 0
#define DEBUG_PERFORMANCE_TRACE 0

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
  audio_buf = new AudioBuffer(SEGMENT_LENGTH * (SEGMENT_NUMBER + 1), SEGMENT_LENGTH, RECORD_SAMPLES);

  messageQueue = xQueueCreate(10, sizeof(uint8_t));
  audioQueue = xQueueCreate(10, sizeof(uint8_t));

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
#if DEBUG_AUDIO_PERFORMANCE == 1
    unsigned long start = millis();
#endif
    audio_buf->stopFillSegment();
    if (initialSegmentCount < SEGMENT_NUMBER)
    {
      initialSegmentCount++;
      continue;
    }
    uint8_t sig = 1;
    xQueueSend(audioQueue, (void *)&sig, portMAX_DELAY);
#if DEBUG_AUDIO_PERFORMANCE == 1
    Serial.print("Audio took ");
    Serial.print(millis() - start);
    Serial.println(" milliseconds");
#endif
  }
}

void process(void *pvParameters)
{

  uint8_t sig;
  for (;;)
  {
    if (xQueueReceive(audioQueue, &sig, portMAX_DELAY) != pdTRUE)
    {
      continue;
    }
#if DEBUG_PERFORMANCE_TRACE == 1
    Serial.print("Start: ");
    Serial.println(millis());
#endif

#if DEBUG_PERFORMANCE == 1
    unsigned long start = millis();
#endif
    int32_t *window = audio_buf->getLatestWindow();

    mfcc(window, mfccMatrix);

#if DEBUG_PERFORMANCE == 1
    unsigned long preprocessing_duration = millis() - start;
#endif
#if DEBUG_PERFORMANCE_TRACE == 1
    Serial.print("Preprocessing: ");
    Serial.println(millis());
#endif

    int8_t label_pos = infer(mfccMatrix);

    if (label_pos == -1)
    {
      continue;
    }

    const char *label = available_classes[label_pos];

#if DEBUG_PERFORMANCE == 1
    unsigned long classification_duration = millis() - start;
#endif
#if DEBUG_PERFORMANCE_TRACE == 1
    Serial.print("Classification: ");
    Serial.println(millis());
#endif

    xQueueSend(messageQueue, (void *)&label_pos, portMAX_DELAY);

#if DEBUG_PERFORMANCE == 1
    unsigned long full_duration = millis() - start;
#endif

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
#if DEBUG_PERFORMANCE_TRACE == 1
    Serial.print("Full: ");
    Serial.println(millis());
#endif
    delay(1);
  }
}
