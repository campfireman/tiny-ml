#ifndef TENSORFLOW_LITE_MODEL_H_
#define TENSORFLOW_LITE_MODEL_H_

// Classes that can be detected by the neural network
extern const char available_classes[][10];
extern const int available_classes_num;

// Pre-trained neural network
alignas(16) extern const unsigned char model[];
extern const int model_len;

#endif /* TENSORFLOW_LITE_MODEL_H_ */
