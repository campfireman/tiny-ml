#include <Arduino.h>
#include "edge-impulse-sdk/dsp/speechpy/feature.hpp"
#include "edge-impulse-sdk/dsp/ei_vector.h"
#include "edge-impulse-sdk/dsp/returntypes.hpp"
#include "edge-impulse-sdk/dsp/ei_vector.h"

#include "preprocessing.hpp"
#include "raw_data.h"

#define GAIN 20

void mfcc(int32_t input[RECORD_SAMPLES], float output[27][NUM_MFCC_COEFFS])
{
    // 1. Create the signal struct:
    ei::signal_t signal;
    signal.total_length = RECORD_SAMPLES;
    signal.get_data = [input](size_t offset, size_t length, float *out_ptr)
    {
        // rawBuf holds your int32_t samples >>16 → int16_t
        for (size_t i = 0; i < length; i++)
        {
            // convert each sample to float in [–1, 1]
            int16_t s = (input[offset + i] * GAIN) >> 16;
            // int16_t s = features[offset + i];
            out_ptr[i] = float(s) / 32768.0f;
        }
        return ei::EIDSP_OK;
    };

    // Construct in one shot:
    ei::matrix_t mfcc_out(27,
                          NUM_MFCC_COEFFS,
                          &output[0][0]);

    int32_t status = ei::speechpy::feature::mfcc(
        &mfcc_out,
        &signal,
        SAMPLE_RATE,                      // 16000
        float(FRAME_SIZE) / SAMPLE_RATE,  // e.g. 2048/16000 = 0.128 s
        float(HOP_SIZE) / SAMPLE_RATE,    // e.g. 512/16000  = 0.032 s
        /*num_cepstral*/ NUM_MFCC_COEFFS, // e.g. 13
        /*num_filters*/ 40,
        /*fft_length*/ FRAME_SIZE, // must be ≥ frame_length in samples
        /*low_freq*/ 0,
        /*high_freq*/ SAMPLE_RATE / 2,
        /*append_energy*/ true,
        /*lifter*/ 1);
    Serial.println();

    if (status != ei::EIDSP_OK)
    {
        Serial.print("MFCC error, code = ");
        Serial.println(status);
        // Optionally decode into human-readable form:
        switch (status)
        {
        case ei::EIDSP_OK:
            Serial.println("OK");
            break;
        case ei::EIDSP_OUT_OF_MEM:
            Serial.println("OUT_OF_MEM");
            break;
        case ei::EIDSP_OUT_OF_BOUNDS:
            Serial.println("MATRIX_OUT_OF_BOUNDS");
            break;
        case ei::EIDSP_PARAMETER_INVALID:
            Serial.println("NOT_VALID_PARAM");
            break;
        // … add other EIDSP_* codes from dsp/returntypes.hpp …
        default:
            Serial.println("UNKNOWN_ERROR");
            break;
        }
    }
}