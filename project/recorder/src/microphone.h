#include <stdint.h>

void microphoneInit(uint32_t samplesRate);
uint32_t microphoneListen(int32_t *rawBuf, int32_t noSamples);