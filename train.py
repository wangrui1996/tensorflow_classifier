import yaml
import argparse
from libs.datatool import ImageData

from tensorflow.python import keras
from libs.utils import get_model_by_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file', default='./configs/config_ms1m_100.yaml')
    return parser.parse_args()


def train(config):

    train_data =  ImageData(config)
    test_data = ImageData(config)
    model = get_model_by_config(config)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=config["lr"]),
                  merics=["accuracy"])
    model.fit_generator(
        train_data.generator(config["train_dir"]),
        steps_per_epoch=config["data_size"] // config["batch_size"],
        epochs=config["epochs"],
        validation_data=test_data.generator(config["test_dir"]),
        validation_steps=config["validation_steps"])

    model.save(config["model_path"])
    model.save_weights(config["weights_path"])

def test(config):
    test_data = ImageData(config)
    model = get_model_by_config(config)
    model.evaluate()


if __name__ == '__main__':
    args = parse_args()
    config = yaml.load(open(args.config_path))

