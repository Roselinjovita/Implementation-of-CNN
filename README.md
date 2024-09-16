# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset
Develop a model that can classify images of handwritten digits (0-9) from the MNIST dataset with high accuracy. The model should use a convolutional neural network architecture and be optimized using early stopping to avoid overfitting.

## Neural Network Model

![exp2 img DL](https://github.com/user-attachments/assets/bb6e3297-8497-47b9-a7fc-042623ec5fbb)


## DESIGN STEPS

### STEP 1:
Import the necessary packages

### STEP 2:
Load the dataset and inspect the shape of the dataset

### STEP 3:
Reshape and normalize the images

### STEP 4:
Use EarlyStoppingCallback function

### STEP 5:
Create and compile the model

### STEP 6:
Get the summary of the model

## PROGRAM

### Name: ROSELIN MARY JOVITA.S
### Register Number: 212222230122
```
import numpy as np
import tensorflow as tf

# Provide path to get the full path
data_path ='/content/mnist.npz.zip'

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")


# reshape_and_normalize

def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """

    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

    # Normalize pixel values
    images = images / 255.0

    ### END CODE HERE ###

    return images

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)
print('Name: S.ROSELIN MARY JOVITA           RegisterNumber: 212222230122          \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")


# EarlyStoppingCallback

### START CODE HERE ###

# Remember to inherit from the correct class
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self, epoch, logs=None):

        # Check if the accuracy is greater or equal to 0.995
        if logs.get('accuracy') >= 0.995:

            # Stop training once the above condition is met
             self.model.stop_training = True

             print("\nReached 99.5% accuracy so cancelling training!\n")
             print('Name: ROSELIN MARY JOVITA.S          Register Number: 212222230122       \n')
### END CODE HERE ###


# convolutional_model

def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    ## START CODE HERE ###

    # Define the model
    model = tf.keras.models.Sequential([
        # First Convolutional Layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second Convolutional Layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten the output from the previous layers
        tf.keras.layers.Flatten(),

        # Fully Connected Layer
        tf.keras.layers.Dense(128, activation='relu'),

        # Output Layer
        tf.keras.layers.Dense(10, activation='softmax') # Assuming 10 classes for output
    ])

    ### END CODE HERE ###

    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

    return model

# Define your compiled (but untrained) model
model = convolutional_model()

training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])

model.summary()
```

## OUTPUT

### Reshape and Normalize output
![Screenshot 2024-09-16 114457](https://github.com/user-attachments/assets/6b1c5180-da92-492a-8916-638562936de2)



### Training the model output

![Screenshot 2024-09-16 114516](https://github.com/user-attachments/assets/3c7efe42-ea09-4d94-ba20-3dd4f70e18f1)

### Summary of the model

![Screenshot 2024-09-16 114525](https://github.com/user-attachments/assets/742d63f5-a65b-479a-a8e1-f1fcd6ac2a60)




## RESULT

Thus the program to create a Convolution Neural Network to classify images is successfully implemented.
