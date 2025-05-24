#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "raw_data.h"
#include "main_functions.h"
#include "gesture_model.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;

    constexpr int kTensorArenaSize = 128 * 1024; // 128 KB
    uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// The name of this function is important for Arduino compatibility.
void setup()
{
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(gesture_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        MicroPrintf("Model provided is schema version %d not equal to supported "
                    "version %d.",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Pull in only the operation implementations we need.
    static tflite::MicroMutableOpResolver<13> resolver;
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddExpandDims();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddMaxPool2D();
    resolver.AddBatchToSpaceNd();
    resolver.AddSpaceToBatchNd();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    MicroPrintf("Attempting AllocateTensors()...");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        MicroPrintf("AllocateTensors() failed");
        MicroPrintf("Arena used (if any): %d", interpreter->arena_used_bytes());
        return;
    }

    MicroPrintf("Tensors allocated. Arena used: %d bytes", interpreter->arena_used_bytes());

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    MicroPrintf("Input tensor has %d bytes", input->bytes);
    MicroPrintf("Input type = %d", input->type);
    MicroPrintf("Input dims: %d", input->dims);
    MicroPrintf("Input dims: %d", input->dims->size);
    for (int i = 0; i < input->dims->size; ++i)
    {
        MicroPrintf("dim[%d] = %d", i, input->dims->data[i]);
    }
}

// The name of this function is important for Arduino compatibility.
void loop()
{
    for (int i = 0; i < 200; i++)
    {
        input->data.f[i * 3 + 0] = features[i];
        input->data.f[i * 3 + 1] = features[i + 1];
        input->data.f[i * 3 + 2] = features[i + 2];
    }
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk)
    {
        MicroPrintf("Invoke failed on x: \n");
        return;
    }
    MicroPrintf("Output type = %d", output->type);

    for (int i = 0; i < available_classes_num; i++)
    {

        MicroPrintf("%s: %f", available_classes[i], output->data.f[i]);
    }
}