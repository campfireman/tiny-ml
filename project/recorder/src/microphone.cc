#include "microphone.h"

#include <Arduino.h>
#include <driver/i2s.h>

#define SAMPLE_BUFFER_SIZE 1024
#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define GAIN 10.0

#define I2S_MIC_CHANNEL I2S_CHANNEL_FMT_ONLY_LEFT
#define I2S_MIC_SERIAL_CLOCK GPIO_NUM_10
#define I2S_MIC_LEFT_RIGHT_CLOCK GPIO_NUM_17
#define I2S_MIC_SERIAL_DATA GPIO_NUM_18

i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_MIC_CHANNEL,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = SAMPLE_BUFFER_SIZE,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0};

i2s_pin_config_t i2s_pins = {
    .bck_io_num = I2S_MIC_SERIAL_CLOCK,
    .ws_io_num = I2S_MIC_LEFT_RIGHT_CLOCK,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_MIC_SERIAL_DATA};

void microphoneInit()
{
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, nullptr);
    i2s_set_pin(I2S_NUM_0, &i2s_pins);
}

uint32_t microphoneListen(int32_t *rawBuf, int16_t *resultBuf, int32_t noSamples)
{
    size_t bytesIn = 0;
    i2s_read(I2S_NUM_0, rawBuf, noSamples * sizeof(int32_t), &bytesIn, portMAX_DELAY);
    int samples = bytesIn / sizeof(int32_t);
    uint32_t dataBytes = samples * sizeof(int16_t);

    // send PCM16LE data
    for (int i = 0; i < samples; ++i)
    {
        int32_t amplified = (int32_t)(rawBuf[i] * GAIN);
        int16_t pcm = amplified >> 16; // down-convert
        resultBuf[i] = pcm;
    }
    return dataBytes;
}