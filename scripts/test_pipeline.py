# Updated Test Pipeline Script

import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model from the correct path
model_path = 'extracted_model/hand_landmarks_detector.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Define input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to make predictions
def make_prediction(image):
    # Preprocess the image
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(image, (input_shape[2], input_shape[1]))
    image_normalized = np.expand_dims(image_resized, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_normalized)
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Test the model with a sample image
if __name__ == '__main__':
    img = cv2.imread('test_image.jpg')  # Make sure to have a sample image filename
    predictions = make_prediction(img)
    print(predictions)  # Display the predictions