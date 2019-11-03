import os
import argparse
import numpy as np
import math
import yaml
import tensorflow as tf
from tensorflow.python import keras as keras
from datetime import datetime

from libs.datatool import ImageData
from libs.utils import get_model_by_config, check_folders,  get_port

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to config file', default='./configs/config_ms1m_100.yaml')
    return parser.parse_args()

class Trainer:
    def __init__(self, config):
        self.config = config
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.output_dir = os.path.join(config['output_dir'], subdir)
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'log')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.debug_dir = os.path.join(self.output_dir, 'debug')
        check_folders([self.output_dir, self.model_dir, self.log_dir, self.checkpoint_dir, self.debug_dir])
        self.val_log = os.path.join(self.output_dir, 'val_log.txt')

        self.batch_size = config['batch_size']
        self.gpu_num = config['gpu_num']
        if self.batch_size % self.gpu_num != 0:
            raise ValueError('batch_size must be a multiple of gpu_num')
        self.image_size = config['image_size']
        self.epoch_num = config['epoch_num']
        self.step_per_epoch = config['step_per_epoch']
        self.val_freq = config['val_freq']
        self.val_data = config['val_data']

        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(self.config))


    def build(self, batch_size):
        cid = ImageData(img_size=self.image_size, augment_flag=self.config['augment_flag'], augment_margin=self.config['augment_margin'])
#        dataset_size = cid.get_dataset_size(self.config["train_data"])
        if self.config["dataset_size"] != None:
            self.step_per_epoch = math.ceil(self.config["dataset_size"]/self.batch_size)
        train_dataset = cid.read_TFRecord(self.config['train_data']).shuffle(10000).repeat().batch(batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        self.train_images, self.train_labels = train_iterator.get_next()
        print("image: ", self.train_images.get_shape())
        print("labels ", self.train_labels.get_shape())

        self.model, self.inference_loss = get_model_by_config([112, 112, 3], self.train_labels, self.train_images, config)

        if self.gpu_num > 1:
            self.model = keras.utils.multi_gpu_model(self.model, gpus=self.gpu_num)
        self.embds = self.model.output

#            self.wd_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#            self.train_loss = self.inference_loss+self.wd_loss

        self.train_op = keras.optimizers.RMSprop()
       # self.model.compile(self.train_op,loss=self.inference_loss)
        self.model.compile(self.train_op, loss=self.inference_loss, metrics=['acc'])

    def set_tf_config(self, num_workers):
        import json
        tf_config = json.dumps({
            'cluster': {
                'worker': []
            },
            'task': {'type': 'worker', 'index': 0}
        })
        tf_config = json.loads(tf_config)
        for port in get_port(num_workers):
            tf_config["cluster"]["worker"].append("localhost:{}".format(port))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

    def train(self):
        self.build(config["batch_size"])
        self.best_acc = 0
        counter = 0
        outter_class = self
        from tensorflow.python.keras import backend as K
        self.func = K.function([self.model.input], [self.model.output])
        class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print("finished one epoch")

            def on_batch_end(self, batch, logs=None):
                if batch % 1000 == 0:
                    json_config = outter_class.model.to_json()
                    with open(outter_class.config["model_config"], 'w') as json_file:
                        json_file.write(json_config)

                    # Save weights to disk
                    outter_class.model.save_weights(outter_class.config["model_weights"])

        model_weights = self.config["model_weights"]
        if os.path.exists(model_weights):
            self.model.load_weights(model_weights)

        self.model.fit(self.train_images, self.train_labels, batch_size=self.config["batch_size"], epochs=100000,
                       steps_per_epoch=self.step_per_epoch,
                       #steps_per_epoch=2,
                       callbacks=[LossAndErrorPrintingCallback()])

if __name__ == '__main__':
    args = parse_args()
    config = yaml.load(open(args.config_path))
    trainer = Trainer(config)
    trainer.train()


