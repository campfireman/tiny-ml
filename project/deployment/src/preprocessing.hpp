#include <stdint.h>

#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
#define FRAME_SIZE 2048
#define HOP_SIZE 512
#define MAX_FRAMES (1 + (RECORD_SAMPLES - FRAME_SIZE) / HOP_SIZE) - 1
#define NUM_MFCC_COEFFS 13

void mfcc(int32_t input[RECORD_SAMPLES], float output[27][NUM_MFCC_COEFFS]);
