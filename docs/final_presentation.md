---
marp: true
theme: rose-pine-dawn
---

<style type="text/css">
.columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
}

img {
    background-color: transparent !important;
    max-height: 60vh;
    height: auto;
    width: auto;
    max-width: 100%;
}

img[alt~="center"] {
    display: block;
    margin: 0 auto;
}

img[alt~="margin"] {
    margin-top: 2em;
    margin-bottom: 2em;
}
</style>

# TinyML Project: Keyword Spotting

Ture Claussen

---

## Goal

![goal center margin](assets/goal.drawio.png)

⇨ **Always-on, responsive and low power** control of lights in living room

---

## Approach

<div class="columns">
<div>

![arduino](./assets/arduino_nano_esp32.jpg)

</div>

<div>

- **Hardware**: [Arduino Nano ESP32](https://docs.arduino.cc/hardware/nano-esp32/) (dual core, 240Mhz, 320kB SRAM) with external I2S microphone
- **Preprocessing**: 13 Mel-frequency cepstral coefficients (MFCCs) via Edge Impulse SDK
- **Optimization**: Quantization, Tensorflow Optimization

</div>
</div>

---

## Data Collection

<div class="columns">
<div>

### Raw

- sunblast: 600s
- chillaxo: 600s
- goodnight: 600s
- zenmode: 602s
- idle: 616s
- unknown: 912s

➡️ Total: 3930s (1.09h)

</div>
<div>

### Augmentation

- Noise
- Pitch
- Shift
- Splice out

➡️ x4 More data
➡️ Total: 15720s (4.37h)

</div>

---

## Model Architecture

![architecture center margin](assets/architecture.drawio.png)

---

## Model Properties

- Total params: 274,132
- Trainable params: 91,270
- Accuracy: 0.8379
- Quantized accuracy: 0.8354

---

## Confusion Matrix

![confusion center margin](assets/confusion_bigger_model.png)

---

## Technical Overview

![overview center margin](assets/technical_overview.drawio.png)

---

## Continuous Audio via Ring Buffer

![ring center margin](assets/ring_buffer.drawio.png)

---

## Performance

![performance center margin](assets/performance.drawio.png)

---

## Memory Footprint

- Total SRAM Allocated: 98kB (33%)
- Total SRAM Free: 300kB
- Total PSRAM Allocated: 172kB (2%)
- Total PSRAM Free: 8,385kB

---

## Demo Time!

---

## Project Result

1. **Phase: MVP (5 points)**
   - Self-created, small dataset with multiple human voices ✅
   - Detection of one command spoken clearly, close and without noise ✅
   - Receive classification result via serial monitor ✅
2. **Phase: Extended MVP (10 points)**
   - Addtion of three more commands (max three syllables) ✅
   - Extension of dataset with synthetic data (with TTS model) ✅
3. **Phase: Usable Product (15 points)**
   - Integration: Works with MQTT and Home Assistant ✅
   - Reactivity: Time from command to MQTT message in less than one second ✅
   - Robustness: All distances and usual noice scenarios (conversation, TV etc.) ➖

---

## Learnings

- Tensorflow Optmization (QAT etc.) only supports Keras 2 and has very limited architecture support
- Synthetic data did not add as much value as hoped

---

## Future Work

- Get rid of dynamic memory allocations
- Optimize Fourier Transformation even more (radix 4)
- More data augmentation
- More advanced architectures
- NAS is promising as hyperparameters are quite random at the moment
