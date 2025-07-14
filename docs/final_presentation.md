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

# TinyML Project: Keyword Spotting

Ture Claussen

---

## Goal

![goal center margin](assets/goal.drawio.png)

⇨ **Always-on, responsive and low power** control of lights in living room

---

## Technical Overview

![overview center margin](assets/technical_overview.drawio.png)

---

## Ring Buffer

![ring center margin](assets/ring_buffer.drawio.png)

---

## Project Result

1. **Phase: MVP (5 points)** ✅
   - Self-created, small dataset with multiple human voices
   - Detection of one command spoken clearly, close and without noise
   - Receive classification result via serial monitor
2. **Phase: Extended MVP (10 points)**
   - Addtion of three more commands (max three syllables)
   - Extension of dataset with synthetic data (with TTS model)
3. **Phase: Usable Product (15 points)**
   - Integration: Works with MQTT and Home Assistant ✅
   - Reactivity: Time from command to MQTT message in less than one second ✅
   - Robustness: All distances and usual noice scenarios (conversation, TV etc.) ✅
