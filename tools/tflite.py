import cv2
import tensorflow as tf
import numpy as np
tflite_model = "./demo.tflite"
image_path = "demo.jpg"
# Load TFLite model and allocate tensors.

class classfier():
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # Test the TensorFlow Lite model on random input data.
        input_shape = self.input_details[0]['shape']
        self.input_width = input_shape[-2]
        self.input_height = input_shape[-3]
        self.input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)


    def progress(self, image):
        image = cv2.resize(image, (self.input_width, self.input_height))
        self.input_data[0] = image
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)


        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        tflite_results = self.interpreter.get_tensor(self.output_details[0]['index'])
        max_score = -1
        max_idx = -1
        for idx, score in enumerate(tflite_results[0]):
            print(idx, score)
            if max_score < score:
                max_score =score
                max_idx = idx
        return max_idx

classfier = classfier(tflite_model)

img = cv2.imread("demo.jpg")
img = cv2.resize(img, (112,112))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 127.5 - 1
print(classfier.progress(img))