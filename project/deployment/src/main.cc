#include <Arduino.h>
#include <Arduino_LSM9DS1.h>

#include "inference.hpp"
#include "microphone.h"
#include "model.h"
#include "mqtt.h"
#include "preprocessing.hpp"
#include "wifi.hpp"

#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define NUM_MFCC_COEFFS 13
#define CORE_0 0
#define CORE_1 1

static QueueHandle_t commandQueue;

void process(void *pvParameters);
void message(void *pvParameters);

int32_t rawBuf[RECORD_SAMPLES];
float mfccMatrix[32][NUM_MFCC_COEFFS];

void setup()
{
  Serial.begin(115200);

  microphoneInit(SAMPLE_RATE);
  inferenceInit();
  wifiInit();
  mqttInit();

  commandQueue = xQueueCreate(10, sizeof(u_int8_t));

  if (commandQueue == NULL)
  {
    Serial.println("Failed to create command queue.");
    for (;;)
      ;
  }

  xTaskCreatePinnedToCore(
      process,
      "Audio processing",
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

void process(void *pvParameters)
{

  for (;;)
  {
    Serial.println("rec");
    uint32_t n = microphoneListen(rawBuf, RECORD_SAMPLES);
    Serial.println("st");

    unsigned long start = millis();

    mfcc(rawBuf, mfccMatrix);

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
      Serial.println("Sent command.");
    }
  }
}
