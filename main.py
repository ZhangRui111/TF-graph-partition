import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim
import time

from data import Cifar10
from utils import save_cfg_to_yaml, load_cfg_from_yaml, merge_cfg_from_args
from model import resnet_v2


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--policy-lr", type=float, default=1e-5, help="learning rate")
    args = parser.parse_args()

    config_dict = load_cfg_from_yaml('config/CIFAR10/R_50.yaml')
    config_dict = merge_cfg_from_args(config_dict, args)
    lr = config_dict["policy_lr"]
    BATCH_SIZE = config_dict['TRAIN']['BATCH_SIZE']
    MAX_EPOCH = config_dict['TRAIN']['MAX_EPOCH']
    TEST_BATCH = config_dict['TEST']['BATCH_SIZE']

    with tf.Session() as sess:
        # input for resnet
        tf_x = tf.placeholder('float', [None, 32, 32, 3])
        tf_y = tf.placeholder('float', [None, 10])
        # output of resnet
        # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        resnet_out, end_points = resnet_v2.resnet_v2_50(tf_x, 10, is_training=False)

        # loss = tf.reduce_mean(-tf.reduce_sum(tf_y * tf.log(resnet_out)))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=resnet_out, labels=tf_y))
        tf.summary.scalar('loss', loss)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(resnet_out, 1), tf.argmax(tf_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # separate tensorboard
        merge_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)

        # get cifar10 dataset
        cifar10_data = Cifar10('CIFAR10/', config=config_dict)
        test_images, test_labels = cifar10_data.test_data()

        # training
        start_time = time.time()
        batch_counter = 0
        while cifar10_data.epochs_completed < MAX_EPOCH:
            batch_xs, batch_ys = cifar10_data.next_train_batch()
            batch_counter += 1
            sess.run(train_step, feed_dict={tf_x: batch_xs, tf_y: batch_ys})

            if batch_counter % 100 == 0:
                # calculate the train_accuracy:
                train_accuracy = accuracy.eval(feed_dict={tf_x: batch_xs, tf_y: batch_ys})
                print("epoch {} batch {}, training accuracy {}".
                      format(cifar10_data.epochs_completed, batch_counter, train_accuracy))

                result = sess.run(merge_op, {tf_x: batch_xs, tf_y: batch_ys})
                train_writer.add_summary(result, batch_counter)

                end_time = time.time()
                print('time: ', (end_time - start_time))
                start_time = end_time

            # calculate the test_accuracy
            if batch_counter % 1000 == 0:
                # Test_accuracy
                test_accuracy = 0
                n_batch = int(test_images.shape[0] / TEST_BATCH)
                for j in range(n_batch):
                    test_accuracy += accuracy.eval(
                        feed_dict={tf_x: test_images[j * TEST_BATCH:(j + 1) * TEST_BATCH],
                                   tf_y: test_labels[j * TEST_BATCH:(j + 1) * TEST_BATCH]})
                test_accuracy /= n_batch
                print("test accuracy {}".format(test_accuracy))

                _, result = sess.run(
                    [train_step, merge_op],
                    feed_dict={
                        tf_x: test_images[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],
                        tf_y: test_labels[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]})
                test_writer.add_summary(result, batch_counter)

        # Overall test accuracy
        test_accuracy = 0
        n_batch = int(test_images.shape[0] / TEST_BATCH)
        for j in range(n_batch):
            test_accuracy += accuracy.eval(
                feed_dict={tf_x: test_images[j * TEST_BATCH:(j + 1) * TEST_BATCH],
                           tf_y: test_labels[j * TEST_BATCH:(j + 1) * TEST_BATCH]})
        test_accuracy /= n_batch
        print("test accuracy {}".format(test_accuracy))

        save_cfg_to_yaml(config_dict, 'logs/current_config.yaml')


if __name__ == '__main__':
    main()
