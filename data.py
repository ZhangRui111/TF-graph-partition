import numpy as np
import os
import tensorflow as tf


def extract_data(filenames, config):
    """
    extract dataset.
    :param filenames:
    :return:
    """
    LOAD_NUM = config['DATASET']['LOAD_NUM']
    IMAGE_SIZE = config['DATASET']['IMAGE_SIZE']
    NUM_CHANNELS = config['DATASET']['NUM_CHANNELS']
    LABEL_SIZE = config['DATASET']['LABEL_SIZE']
    PIXEL_DEPTH = config['DATASET']['PIXEL_DEPTH']

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    labels = None
    images = None
    init_flag = True

    for f in filenames:
        bytestream = open(f, 'rb')
        buf = bytestream.read(LOAD_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + LABEL_SIZE))
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(LOAD_NUM, LABEL_SIZE + IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)  # data的每个图像，label在前，image在后
        labels_images = np.hsplit(data, [LABEL_SIZE])

        label = labels_images[0].reshape(LOAD_NUM)
        image = labels_images[1].reshape(LOAD_NUM, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

        if init_flag:
            labels = label
            images = image
            init_flag = False
        else:
            labels = np.concatenate((labels, label))
            images = np.concatenate((images, image))
        pass
    # image pre-processing
    images = (images - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

    return labels, images


def extract_train_data(files_dir, config):
    """
    extract train set.
    :param files_dir:
    :return:
    """
    filenames = [os.path.join(files_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    return extract_data(filenames, config)


def extract_test_data(files_dir, config):
    """
    extract test set.
    :param files_dir:
    :return:
    """
    filenames = [os.path.join(files_dir, 'test_batch.bin')]
    return extract_data(filenames, config)


def dense_to_one_hot(labels_dense, num_classes):
    """
    convert dense label to one-hot label.
    [1, 5, ...] --> [[0, 1, 0, ...], [...], ...]
    :param labels_dense:
    :param num_classes:
    :return:
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class Cifar10(object):
    def __init__(self, data_dir, config):
        super(Cifar10, self).__init__()

        self.NUM_CLASSES = config['DATASET']['NUM_CLASSES']
        self.TRAIN_NUMS = config['DATASET']['TRAIN_NUMS']
        self.BATCH_SIZE = config['TRAIN']['BATCH_SIZE']

        self.train_labels, self.train_images = extract_train_data(
            os.path.join(data_dir, 'cifar-10-batches-bin'), config)
        self.test_labels, self.test_images = extract_test_data(
            os.path.join(data_dir, 'cifar-10-batches-bin'), config)

        self.train_labels = dense_to_one_hot(self.train_labels, self.NUM_CLASSES)
        self.test_labels = dense_to_one_hot(self.test_labels, self.NUM_CLASSES)

        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_train_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += self.BATCH_SIZE
        if self.index_in_epoch > self.TRAIN_NUMS:
            self.epochs_completed += 1
            # Reshuffle the train set
            perm = np.arange(self.TRAIN_NUMS)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]
            start = 0
            self.index_in_epoch = self.BATCH_SIZE
            assert self.BATCH_SIZE <= self.TRAIN_NUMS
        end = self.index_in_epoch
        return self.train_images[start:end], self.train_labels[start:end]

    def test_data(self):
        return self.test_images, self.test_labels


# def main():
#     config_dict = load_cfg_from_yaml('config/CIFAR10/R_50.yaml')
#     train_labels, train_images = extract_train_data('CIFAR10/cifar-10-batches-bin', config_dict)
#     print(train_images.shape)
#     c10 = Cifar10('CIFAR10/', config=config_dict)
#     c10.next_train_batch()
#
#
# if __name__ == '__main__':
#     main()
