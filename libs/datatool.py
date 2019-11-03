import os
import signal
import tensorflow as tf
from scipy import misc
import numpy as np
import math
import cv2
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

lock = multiprocessing.Lock()
run = True


def handler(signum, frame):
    print("program is be close by user")
    with lock:
        global done
        done = True
    exit(0)



signal.signal(signal.SIGINT, handler)
cnt = 0
done = False

def augmentation(image, aug_img_size):
    print("starting to add dataset augmentation progress...")
    ori_image_shape = tf.shape(image)
    print("random flip image")
    image = tf.image.random_flip_left_right(image)
    print("random crop image")
    image = tf.image.resize_images(image, [aug_img_size, aug_img_size])
    image = tf.random_crop(image, ori_image_shape)
    print("random brightness")
    image = tf.image.random_brightness(image, 0.3)
    return image

def from_folders_add_record(queue, img_size, writer_path, total):
    writer = tf.python_io.TFRecordWriter(writer_path, options=None)
    global done
    while True:
        while True:
            img_info = queue.get(timeout=2)
            with lock:
                if done:
                    writer.close()
                    return 
                else:
                    break
        path, label = img_info[0], img_info[1]
        img = cv2.imread(path)
        if img_size:
            img = cv2.resize(img, (img_size, img_size))
        label = int(label)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shape = img.shape
        tf_features = tf.train.Features(feature={
            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
            "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=list(shape))),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        })
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()

        writer.write(tf_serialized)
        with lock:
            global cnt
            cnt = cnt + 1
            print('%d/%d' % (cnt, total), end='\r')
            if done:
                writer.close()
                return


class ImageData:

    def __init__(self, img_size=112, augment_flag=True, augment_margin=16):
        self.img_size = img_size
        self.augment_flag = augment_flag
        self.augment_margin = augment_margin

    def get_path_label(self, root):
        ids = []
        for _ in os.listdir(root):
            if os.path.isdir(os.path.join(root, _)):
                ids.append(_)
        ids.sort()
        self.cat_num = len(ids)
        id_dict = dict(zip(ids, list(range(self.cat_num))))
        paths = []
        labels = []
        for i in ids:
            cur_dir = os.path.join(root, i)
            fns = os.listdir(cur_dir)
            paths += [os.path.join(cur_dir, fn) for fn in fns]
            labels += [id_dict[i]] * len(fns)
        return paths, labels, ids

    def image_processing(self, img):
        img.set_shape([None, None, 3])
        img = tf.image.resize_images(img, [self.img_size, self.img_size])

        if self.augment_flag:
            augment_size = self.img_size + self.augment_margin
            img = augmentation(img, augment_size)

        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img

    def write_tfrecord_from_folders(self, read_dir, write_dir, thread_num=os.cpu_count()):
        print('write tfrecord from folders...')
        paths, labels, labels_map = self.get_path_label(read_dir)
        with open(os.path.join(write_dir, "label_map.txt"), "w") as f:
            [f.writelines(_) for _ in labels_map]
        assert (len(paths) == len(labels))
        total = len(paths)
        executor = ThreadPoolExecutor(thread_num)
        fs = []

        from multiprocessing import Queue
        import threading
        BUF_SIZE = thread_num
        queue = Queue(BUF_SIZE)

        class ProducerThread(threading.Thread):
            def __init__(self, paths, labels, queue):
                super(ProducerThread, self).__init__()
                self.queue = queue
                self.paths = paths
                self.labels = labels

            def run(self):
                global done
                for path, label in zip(self.paths, self.labels):
                    while self.queue.full():
                        if done:
                            return
                        pass
                    queue.put([path, label])
                while not self.queue.empty():
                    pass
                with lock:
                    done = True

        p = ProducerThread(paths, labels, queue)
        p.start()

        for id in range(thread_num):
            writer_path = os.path.join(write_dir, "train_{}.tfrecord".format(id))
            img_size = self.img_size
            print("Write data into {}".format(writer_path))
            fs.append(
                executor.submit(from_folders_add_record, queue, img_size, writer_path,
                                total))

        [print(f.result()) for f in fs]

        executor.shutdown(wait=True)
        print('done![%d]' % (total))

    def parse_function(self, example_proto):
        dics = {
            'img': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
            'label': tf.FixedLenFeature(shape=(), dtype=tf.int64)
        }
        parsed_example = tf.parse_single_example(example_proto, dics)
        parsed_example['img'] = tf.decode_raw(parsed_example['img'], tf.uint8)
        parsed_example['img'] = tf.reshape(parsed_example['img'], parsed_example['shape'])
        return self.image_processing(parsed_example['img']), parsed_example['label']

    def get_Filenames_to_list(self, tfrecord_dir):
        tfrecord = []
        filenames = os.listdir(tfrecord_dir)
        for fn in filenames:
            if fn.split(".")[-1] == "tfrecord":
                tfrecord.append(os.path.join(tfrecord_dir, fn))
        return tfrecord

    def get_dataset_size(self, tfrecord_dir):
        c = 0
        for fn in self.get_Filenames_to_list(tfrecord_dir):
            for record in tf.python_io.tf_record_iterator(fn):
                c+=1
                if c % 100 == 0:
                    print("dataset current get: {}".format(c))
        return c

    def read_TFRecord(self, tfrecord_dir):
        dataset = tf.data.TFRecordDataset(self.get_Filenames_to_list(tfrecord_dir), buffer_size=256 << 20)
        return dataset.map(self.parse_function, num_parallel_calls=8)