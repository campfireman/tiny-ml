# TinyML Lab 1

## Task 2

### Question 1. Which features/patterns are representative for each of your motions?

- idle: Only miniscule fluctuations in the sensor data, otherwise data readings constant
- circle: Device is moved in a circle motion. All accelerometers oscillate
- left-right: Acceleration on y-axis and x-axis, while z-axis (height acceleration) is calm. Moving object in different orientation in a plane. However, we have gyro acceleration around the z-axis
- up-down: Depending on how cleanly the up and down movement is performed we see oscillations in the magnetometers or not. The accelerometers are slighlty activated

All modes would be expected to only show in the acceleration data, but the device seems to have been rotated during data collection and some movement leaks into the other axis

### Question 2. Your data doesn’t only include movements but also idle data. Why might it bebeneficial to include idle data?

To differentiate between moving and not moving. If there is no label/data for idle our model always outputs a movement, even though the confidence would be very low.

## Task 5

### Question 3. Which settings did you choose in the Time series data input block? Explain what these settings do and why you think that your choices are a good choice. It might be a good idea to look at the patterns present in your data.

The **window size** is the length of one sample. It should be long enough to detect the movement so longer than one wavelength for a typical movement. We chose a window size of 2000ms since the typical movment seems to have a wavelength of about 1000ms so 2000ms should be enough to capture detect it.

The **stride** is the step size in ms at which data from the window is processed. If the step size is larger than the window we ignore data.

The **frequency** of the data is up or down sampled. since we have quite a slow movement, we are not expecting high frequencies so therefore we can downsample the data a bit to 50 Hz to have less input features and therefore less computational load.


## Task 6

### Question 4. What do the different preprocessing blocks do? Briefly explain them. Why should we use preprocessing blocks instead of feeding the data directly into the neural network like using the raw preprocessing block?

- **Spectral Features**: Moves data from time domain into frequency domain
- **Flatten**: "The Flatten block first rescales axes of the signal if value is different than 1. Then statistical analysis is performed on each window, computing between 1 and 8 features for each axis, depending on the number of selected methods."
- **Raw Data**: Does not perform any processing and feeds the raw data directly to the model

In general, preprocessing is much less computationally intense compared to a (D)NN. If there a patterns or features in the model we know beforehand preprocessing can help to reduce the size of the model drastically while maintaining good performance.

### Question 5. Which preprocessing block creates the best features separating the classes? Add screenshots from the feature explorer. What kind of features did the blocks extract?

#### Spectral Features

![](https://writemd.rz.tuhh.de/uploads/4953b12b-8d05-4a47-ab3a-f0f7d16b7c1a.png)

#### Flatten

![](https://writemd.rz.tuhh.de/uploads/1dd7605b-4a3e-430e-9033-2a347649bb14.png)

#### Raw Data

![](https://writemd.rz.tuhh.de/uploads/af9739a0-4b8c-4e07-90cb-61e4394cd1c6.png)

## Task 7

### Question 6. What are 1D Convolutions? How do they differ from 2D convolutions used for images?

| Aspect             | 1D Convolution                  | 2D Convolution                    |
| ------------------ | ------------------------------- | --------------------------------- |
| Input type         | Sequential data                 | Image or spatial data             |
| Kernel shape       | 1D (e.g., 3)                    | 2D (e.g., 3×3)                    |
| Sliding direction  | Along one axis (e.g., time)     | Along two axes (height and width) |
| Feature extraction | Temporal or sequential features | Spatial features                  |

### Question 7. Briefly explain your models. Is there anything special? Include the code for your models in your submission.

#### Training

- Cycles: 30
- Learned Optimizer: [ ]
- Learning Rate 0.0005

#### Dense

Prepocessing: Spectral

```[python]
model = Sequential()
model.add(Dense(20, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(10, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(classes, name='y_pred', activation='softmax'))
```

#### Dense Wide

Prepocessing: Spectral

```[python]
model = Sequential()
model.add(Dense(64, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(32, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(classes, name='y_pred', activation='softmax'))
```

#### Dense Wide+Deep

Prepocessing: Spectral

```[python]
model = Sequential()
model.add(Dense(128, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(64, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(32, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(classes, name='y_pred', activation='softmax'))
```

#### CNN+Dense Large

Preprocessing: Raw Data

```[python]
model = Sequential([
    Conv1D(16, kernel_size=5, activation='relu', input_shape=(627,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(32, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.3),

    Dense(classes, activation='softmax', name='y_pred')
])
```

#### CNN+Dense Small

Preprocessing: Raw Data

```[python]
model = Sequential([
    Conv1D(8, kernel_size=5, activation='relu', input_shape=(627,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(16, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.3),

    Dense(classes, activation='softmax', name='y_pred')
])
```

#### CNN+Dense Tiny

Preprocessing: Raw Data

```[python]
model = Sequential([
    Conv1D(8, kernel_size=5, activation='relu', input_shape=(627,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.3),

    Dense(classes, activation='softmax', name='y_pred')
])
```

#### CNN

Preprocessing: Raw Data

```[python]
model = Sequential([
    Conv1D(8, kernel_size=5, activation='relu', input_shape=(627,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(16, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Flatten(),

    Dense(classes, activation='softmax', name='y_pred')
])
```

#### CNN Small

Preprocessing: Raw Data

```[python]
model = Sequential([
    Conv1D(4, kernel_size=5, activation='relu', input_shape=(627,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(8, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Flatten(),

    Dense(classes, activation='softmax', name='y_pred')
])
```

#### CNN Tiny

Preprocessing: Raw Data

```[python]
model = Sequential([
    Conv1D(16, kernel_size=5, activation='relu', input_shape=(627,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Flatten(),

    Dense(classes, activation='softmax', name='y_pred')
])
```

#### CNN Micro

Preprocessing: Raw Data

```[python]
model = Sequential([
    Conv1D(8, kernel_size=5, activation='relu', input_shape=(627,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Flatten(),

    Dense(classes, activation='softmax', name='y_pred')
])
```



### Question 8. What is the training performance of the models? How much memory is your model expected to need? What execution time is estimated? You can add screenshots. Check also for the memory usage and execution time of your DSP blocks.

#### Dense
![](https://writemd.rz.tuhh.de/uploads/8459311b-20a0-4dfc-9386-099912f5239e.png)

#### Dense Wide

![](https://writemd.rz.tuhh.de/uploads/36eca07e-39da-4c45-8c59-0cea7f975a1e.png)

#### Dense Wide+Deep

![](https://writemd.rz.tuhh.de/uploads/8137a3fb-7240-4afa-9274-25af97374890.png)

#### CNN+Dense Large

![](https://writemd.rz.tuhh.de/uploads/ec5b3cc0-71ea-43d3-a2f2-1d4cd97be9ac.png)

#### CNN+Dense Small

![](https://writemd.rz.tuhh.de/uploads/5243afe6-2c6e-42dc-98d9-3550d4bcfd19.png)

#### CNN+Dense Tiny

![](https://writemd.rz.tuhh.de/uploads/529961a7-e4db-448e-af71-dda38646ae8e.png)

#### CNN

![](https://writemd.rz.tuhh.de/uploads/59761836-c531-4f77-ad8b-4117cf2473c5.png)

#### CNN Small

![](https://writemd.rz.tuhh.de/uploads/ddb9528b-538b-4030-a1fc-81456217a9f3.png)

#### CNN Tiny

![](https://writemd.rz.tuhh.de/uploads/63073aea-fc62-42e9-86d5-5a49842f0200.png)

#### CNN Micro

Despite really good training accuracy of 98%! -> Bad generalization

![](https://writemd.rz.tuhh.de/uploads/c388415a-59d8-4e0a-8e7e-c4b56338fa85.png)

## Task 8

### Question 9. How good do your models classify the test data? Do the models generalize well? Create plots (bar graphs) comparing your models. Also explain F1 Score and Uncertainty displayed in the confusion matrix.

![](https://writemd.rz.tuhh.de/uploads/bd2ea52a-25ce-400a-9616-a0efdbae1a07.png)

![](https://writemd.rz.tuhh.de/uploads/abda6b77-1319-4f61-a5ee-94842863bcac.png)

- **F1-Score**: "The F1 score is the harmonic mean of the precision and recall. It thus symmetrically represents both precision and recall in one metric."
- **Uncertainty**: Removed from confusion matrix


| Model           | Preprocessing | Accuracy (%) | Inference Time (ms) | RAM usage (bytes) | Flash Usage (bytes) |
| --------------- | ------------- | ------------ | ------------------- | ----------------- | ------------------- |
| Dense           | spectral      | 98.8         | 1                   | 1.4K              | 16.1K               |
| Dense Wide      | spectral      | 99.4         | 1                   | 1.4K              | 19.9K               |
| Dense Wide+Deep | spectral      | 99.0         | 1                   | 1.7K              | 31K                 |
| Dense EON       | spectral      | 97.3         | 1                   | 1.7K              | 20.8K               |
| CNN+Dense Large | raw           | 100          | 56                  | 23.7K             | 210K                |
| CNN+Dense Small | raw           | 100          | 44                  | 15.8K             | 93.1K               |
| CNN+Dense Tiny  | raw           | 99.2         | 45                  | 12K               | 90.1K               |
| CNN             | raw           | 100          | 45                  | 15.6K             | 64.1K               |
| CNN Small       | raw           | 99.8         | 34                  | 11.6K             | 58.6K               |
| CNN Tiny        | raw           | 98.2         | 28                  | 18.8K             | 70.6K               |
| CNN Micro       | raw           | 94.9         | 32                  | 11.9K             | 60.7K               |

## Task 10

### Question 10. Did the EON Tuner come up with a better model than you? If so, in which regard is it better? Is it still better when you limit it to using only accelerometer data? (To answer the latter question, first answer Question 11.)

Nice separation of features

![](https://writemd.rz.tuhh.de/uploads/44be0efd-3de3-4fb4-8e15-3008bb085cfe.png)

```[python]
model = Sequential()
model.add(Dense(40, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(20, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(10, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dropout(0.5))
model.add(Dense(classes, name='y_pred', activation='softmax'))
```

![](https://writemd.rz.tuhh.de/uploads/ab1e751b-f6f1-4689-ad71-ca3594771fb6.png)

After removal of all features except accelerometer data it performs worse:

![](https://writemd.rz.tuhh.de/uploads/46b827e4-d19b-46e8-90c1-c90444948d6c.png)

### Question 11. If the EON tuner resulted in a better model, add this model as a new impulse and use it as well in the Model Testing section, and add its results to the plots you created in Question 9. Please note: Before doing so, save your current version with the versioning tool on the left.

## Task 12

### Question 12. Explain the output of the classification results. Does your model work? Did it misclassify some timestamps? Is this a bad thing? Why (not)?

## Task 13

### Question 13. Does the classification also work for your phone? If it doesn’t work, why not? Does the performance change when you change the orientation of your phone? In which orientation do you have to hold your phone for it to work best?

#### First Run

- Phone orientation: Flat
- Movement: Circle
- Classification: Up-Down

![](https://writemd.rz.tuhh.de/uploads/c7066104-555e-4467-abb9-d90dd2eabeec.png)


#### Second Run

- Phone orientation: Upright
- Movement: Circle
- Classification: Left-Right

![](https://writemd.rz.tuhh.de/uploads/a22f1640-30ef-4a15-8484-c28e31847c99.png)


#### Third Run

- Phone orientation: Flat
- Movement: Up-Down
- Classification: Idle

![](https://writemd.rz.tuhh.de/uploads/5e901e2a-7728-4b3a-8a16-a1c0b393ef2b.png)

#### Fourth Run

- Phone orientation: Upright
- Movement: Up-Down
- Classification: Left-Right

![](https://writemd.rz.tuhh.de/uploads/041e27a3-041b-4018-9505-0c0d183d7baf.png)

#### Fifth Run

- Phone orientation: Flat
- Movement: Left-Right
- Classification: Left-Right

![](https://writemd.rz.tuhh.de/uploads/4037aac4-1190-4f3a-96f7-875668ae6baa.png)


#### Sixth Run

- Phone orientation: Flat, upside-down
- Movement: Up-Down
- Classification: Up-Down

![](https://writemd.rz.tuhh.de/uploads/35ed22e4-bff1-49ab-b762-e291e1942318.png)

#### Seventh Run

- Phone orientation: Flat, upside-down
- Movement: None
- Classification: Idle

![](https://writemd.rz.tuhh.de/uploads/d5840f99-f3f4-42d0-983f-5996c081ec01.png)
