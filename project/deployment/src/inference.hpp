#include <stdint.h>

#define NUM_MFCC_COEFFS 13

void inferenceInit();
uint8_t infer(float mfccMatrix[32][NUM_MFCC_COEFFS]);
