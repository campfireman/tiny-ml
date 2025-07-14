#include <Arduino.h>
#include <driver/i2s.h>

#include "microphone.h"
#include "audio_buffer.hpp"

#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define SEGMENT_NUMBER 4
#define SEGMENT_LENGTH (RECORD_SAMPLES / SEGMENT_NUMBER)
#define GAIN 20

AudioBuffer *audio_buf;

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
    microphoneInit(SAMPLE_RATE);
    Serial.println("Ready. Send 'r' to record.");
    audio_buf = new AudioBuffer(SEGMENT_LENGTH * (SEGMENT_NUMBER + 1), SEGMENT_LENGTH, RECORD_SAMPLES);
}

void loop()
{
    microphoneListen(audio_buf->startFillSegment(), SEGMENT_LENGTH);
    audio_buf->stopFillSegment();
    if (Serial.available() && Serial.read() == 'r')
    {
        int segmentCount = 0;
        uint32_t samples = 0;
        digitalWrite(LED_BUILTIN, HIGH);
        for (;;)
        {
            samples += microphoneListen(audio_buf->startFillSegment(), SEGMENT_LENGTH);
            audio_buf->stopFillSegment();
            if (segmentCount < SEGMENT_NUMBER)
            {
                segmentCount++;
                continue;
            }
            break;
        }
        digitalWrite(LED_BUILTIN, LOW);
        int32_t *windowPtr = audio_buf->getLatestWindow();

        writeWavHeader(samples * sizeof(int16_t));

        for (int i = 0; i < samples; ++i)
        {
            uint16_t pcm = (windowPtr[i] * GAIN) >> 16;
            Serial.write((uint8_t *)&pcm, 2);
        }
        Serial.println(); // flush
    }
}
