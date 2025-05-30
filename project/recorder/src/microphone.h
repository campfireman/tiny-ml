#include <stdint.h>

void microphoneInit();
uint32_t microphoneListen(int32_t *rawBuf, int16_t *resultBuf, int32_t noSamples);