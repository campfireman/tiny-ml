#include <edge-impulse-sdk/tensorflow/lite/micro/all_ops_resolver.h>
#include <edge-impulse-sdk/tensorflow/lite/micro/micro_interpreter.h>
#include <edge-impulse-sdk/tensorflow/lite/schema/schema_generated.h>
#include <Arduino.h>

#include "inference.hpp"
#include "log.hpp"
#include "model.h"

#define DEBUG_CLASSIFICATION 0
#define CONFIDENCE_THRESHOLD 50

tflite::AllOpsResolver tflOpsResolver;
const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;

float zeroPoint = 0.0;
float scale = 0.0;

constexpr int tensorArenaSize = 32 * 1024;

byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

inline int8_t quantize_int8(float x, float scale, int zero_point)
{
    int q = static_cast<int>(std::lround(x / scale) + zero_point);
    if (q < -128)
        q = -128;
    if (q > 127)
        q = 127;
    return static_cast<int8_t>(q);
}

void inferenceInit()
{
    tflModel = tflite::GetModel(model);

    if (tflModel->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println("Model schema mismatch!");
        while (1)
            ;
    }

    tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

    if (tflInterpreter->AllocateTensors(true) != kTfLiteOk)
    {
        Serial.println("Allocation failed!");
        while (true)
            ;
    }

    tflInputTensor = tflInterpreter->input(0);
    tflOutputTensor = tflInterpreter->output(0);
    zeroPoint = tflInputTensor->params.zero_point;
    scale = tflInputTensor->params.scale;
}

int8_t infer(float mfccMatrix[32][NUM_MFCC_COEFFS])
{
    // write to model for inference
    int idx = 0;
    for (int i = 0; i < 27; i++)
    {
        for (int j = 0; j < NUM_MFCC_COEFFS; j++)
        {
            float val_f = mfccMatrix[i][j];
            int8_t val_q = quantize_int8(val_f, scale, zeroPoint);
            tflInputTensor->data.int8[idx++] = val_q;
            // Serial.print(val_f, 6);
            // Serial.print(",");
        }
        // Serial.println();
    }

    TfLiteStatus invokeStatus = tflInterpreter->Invoke();

    if (invokeStatus != kTfLiteOk)
    {
        Serial.println("Invoke failed!");
        while (1)
            ;
    }

    int8_t max = INT8_MIN;
    int8_t label_pos;
    for (int i = 0; i < available_classes_num; i++)
    {

        if (tflOutputTensor->data.int8[i] > max)
        {
            max = tflOutputTensor->data.int8[i];
            label_pos = i;
        }
#if DEBUG_CLASSIFICATION == 1
        Serial.print(available_classes[i]);
        Serial.print(": ");
        Serial.println(tflOutputTensor->data.int8[i]);
#endif
    }
    const char *label = available_classes[label_pos];
    if (label != "idle" && label != "unknown" && max < CONFIDENCE_THRESHOLD)
    {
        return -1;
    }
    return label_pos;
}