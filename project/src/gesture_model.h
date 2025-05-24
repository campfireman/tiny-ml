#ifndef TENSORFLOW_LITE_MODEL_H_
#define TENSORFLOW_LITE_MODEL_H_

// Classes that can be detected by the neural network
extern const char available_classes[][11];
extern const int available_classes_num;

// Pre-trained netural network
extern const unsigned char gesture_model[];
extern const int gesture_model_len;

#endif /* TENSORFLOW_LITE_MODEL_H_ */