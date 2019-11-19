from tensorflow.python import keras


class ImageData:
    def __init__(self, config):
        self.config = config
        self.data_gen = keras.preprocessing.image.ImageDataGenerator()

    def generator(self, data_dir):
        config = self.config
        return self.data_gen.flow_from_directory(
            data_dir,
            target_size=(config["input_height"], config["input_width"]),
            batch_size=config["batch_size"],
            class_mode="categorical")


