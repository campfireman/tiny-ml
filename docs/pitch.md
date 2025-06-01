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
    background-color: transparent!important;
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

# TinyML Project Pitch

---

## Motivation

<div class="columns">
<div>

- Smart home with [Home Assistant](https://www.home-assistant.io/) is a long time hobby project already: Server, sensors, light bulbs, MQTT, ...
- Already some experience with custom embendded sensor firmware (integrated [Airgradient One](https://www.airgradient.com/indoor/) with MQTT)
- **Have to get up to reach button for toggling light scences...**

</div>
<div>

![relaxo](./assets/relaxo.png)

</div>
</div>

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

- **Hardware**: [Arduino Nano ESP32](https://docs.arduino.cc/hardware/nano-esp32/) (dual core, 240Mhz, 512kB SRAM) with external I2S microphone
- **Preprocessing**: 13 Mel-frequency cepstral coefficients (MFCCs)
- **Architecture**: Most likely Depthwise Separable Convolutional Neural Network (DS-CNN)
- **Optimization**: Quantization, QAT, pruning, ...

</div>
</div>

---

## Expected Result

1. **Phase: MVP (5 points)**
   - Self-created, small dataset
   - Detection of one command spoken clearly, close and without noise
   - Receive classification result via serial monitor
2. **Phase: Extended MVP (10 points)**
   - Addtion of three more commands (max three syllables)
   - Extension of dataset with synthetic data (with TTS model)
3. **Phase: Usable Product (15 points)**
   - Integration: Works with MQTT and Home Assistant
   - Reactivity: Time from command to MQTT message in less than one second
   - Robustness: All distances and usual noice scenarios (conversation, TV etc.)
