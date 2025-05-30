#include <Arduino.h>
#include <driver/i2s.h>

#include "microphone.h"

#define SAMPLE_BUFFER_SIZE 1024
#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define GAIN 10.0

int32_t rawBuf[RECORD_SAMPLES];
int16_t resultBuf[RECORD_SAMPLES];

// Write a 44-byte WAV header to Serial
void writeWavHeader(uint32_t dataBytes)
{
    auto writeLE = [&](const void *p, size_t n)
    { Serial.write((const uint8_t *)p, n); };
    Serial.write("RIFF", 4);
    uint32_t chunkSize = 36 + dataBytes;
    writeLE(&chunkSize, 4);
    Serial.write("WAVE", 4);
    Serial.write("fmt ", 4);
    uint32_t subChunk1Size = 16;
    writeLE(&subChunk1Size, 4);
    uint16_t audioFormat = 1, channels = 1;
    writeLE(&audioFormat, 2);
    writeLE(&channels, 2);
    uint32_t sampleRate = SAMPLE_RATE;
    writeLE(&sampleRate, 4);
    uint16_t bitsPerSample = 16;
    uint32_t byteRate = sampleRate * channels * bitsPerSample / 8;
    writeLE(&byteRate, 4);
    uint16_t blockAlign = channels * bitsPerSample / 8;
    writeLE(&blockAlign, 2);
    writeLE(&bitsPerSample, 2);
    Serial.write("data", 4);
    writeLE(&dataBytes, 4);
}

void setup()
{
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    microphoneInit();
    Serial.println("Ready. Send 'r' to record.");
}

void loop()
{
    if (Serial.available() && Serial.read() == 'r')
    {
        digitalWrite(LED_BUILTIN, HIGH);
        uint32_t dataBytes = microphoneListen(rawBuf, resultBuf, RECORD_SAMPLES);
        digitalWrite(LED_BUILTIN, LOW);

        // send header
        writeWavHeader(dataBytes);

        // send PCM16LE data
        for (int i = 0; i < dataBytes / sizeof(int16_t); ++i)
        {
            // Serial.println(rawBuf[i]);
            Serial.write((uint8_t *)&resultBuf[i], 2);
        }
        Serial.println(); // flush
    }
}