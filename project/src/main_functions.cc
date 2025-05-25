#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_profiler.h"

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
    static tflite::MicroProfiler profiler;

    constexpr int kTensorArenaSize = 128 * 1024; // 256 KB
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
    static tflite::MicroMutableOpResolver<14> resolver;
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
    resolver.AddRelu();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
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
    MicroPrintf("Input dims: %d", input->dims->size);
    for (int i = 0; i < input->dims->size; ++i)
    {
        MicroPrintf("dim[%d] = %d", i, input->dims->data[i]);
    }
}

inline int8_t quantize_int8(float x, float scale, int zero_point)
{
    int q = static_cast<int>(std::lround(x / scale) + zero_point);
    if (q < -128)
        q = -128;
    if (q > 127)
        q = 127;
    return static_cast<int8_t>(q);
}

// The name of this function is important for Arduino compatibility.
void loop()
{
    MicroPrintf("Arena used: %d / %d bytes",
                interpreter->arena_used_bytes(),
                kTensorArenaSize);
    MicroPrintf("Input tensor has %d bytes", input->bytes);
    MicroPrintf("Input type = %d", input->type);
    MicroPrintf("Output type = %d", output->type);
    MicroPrintf("Input dims: %d", input->dims->size);
    for (int i = 0; i < input->dims->size; ++i)
    {
        MicroPrintf("dim[%d] = %d", i, input->dims->data[i]);
    }
    MicroPrintf("IN scale=%f zero_point=%d",
                input->params.scale, input->params.zero_point);
    MicroPrintf("OUT scale=%f zero_point=%d",
                output->params.scale, output->params.zero_point);
    auto *subgraph = model->subgraphs()->Get(0);
    auto *ops = subgraph->operators();
    auto *codes = model->operator_codes();
    size_t num_tensors = subgraph->tensors()->size();

    for (int i = 0; i < ops->size(); ++i)
    {
        int code_idx = ops->Get(i)->opcode_index();
        int builtin = codes->Get(code_idx)->builtin_code();
        MicroPrintf("Node %2d → %s",
                    i,
                    EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(builtin)));
    }

    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    // float output_scale = output->params.scale;
    // int output_zero_point = output->params.zero_point;
    // Compute the number of floats your model expects:
    // int input_len = input->bytes / sizeof(float); // 2400 bytes / 4 = 600 floats
    int input_len = input->bytes; // 600 bytes / 1 = 600 int8

    // Straight copy:
    for (int idx = 0; idx < input_len; ++idx)
    {
        input->data.int8[idx] =
            quantize_int8(features[idx], input_scale, input_zero_point);
    }

    // (Optional) verify the first few elements match
    for (int i = 590; i < input_len; ++i)
    {
        MicroPrintf("features[%d]=%f  input[%d]=%d", i, features[i], i, input->data.int8[i]);
    }
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();

    // profiler.Log();

    if (invoke_status != kTfLiteOk)
    {
        MicroPrintf("Invoke failed!\n");
        return;
    }

    for (int i = 0; i < available_classes_num; i++)
    {

        MicroPrintf("%s: %d", available_classes[i], output->data.int8[i]);
    }
}