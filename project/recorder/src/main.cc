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

int32_t rawBuf[RECORD_SAMPLES];

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
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, nullptr);
    i2s_set_pin(I2S_NUM_0, &i2s_pins);
    Serial.println("Ready. Send 'r' to record.");
}

void loop()
{
    if (Serial.available() && Serial.read() == 'r')
    {
        size_t bytesIn = 0;
        digitalWrite(LED_BUILTIN, HIGH);
        i2s_read(I2S_NUM_0, rawBuf, RECORD_SAMPLES * sizeof(int32_t), &bytesIn, portMAX_DELAY);
        digitalWrite(LED_BUILTIN, LOW);
        int samples = bytesIn / sizeof(int32_t);
        uint32_t dataBytes = samples * sizeof(int16_t);

        // send header
        writeWavHeader(dataBytes);

        // send PCM16LE data
        for (int i = 0; i < samples; ++i)
        {
            int32_t amplified = (int32_t)(rawBuf[i] * GAIN);
            int16_t pcm = amplified >> 16; // down-convert
            Serial.write((uint8_t *)&pcm, 2);
        }
        Serial.println(); // flush
    }
}