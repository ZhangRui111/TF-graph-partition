import argparse
import numpy as np
import os
import tensorflow.compat.v1 as tf
import tf_slim as slim
from tensorflow.python.client import device_lib
import time

from data import Cifar10
from utils import save_cfg_to_yaml, load_cfg_from_yaml, merge_cfg_from_args, data_type, next_device
from model import resnet_v2

# os.environ["CUDA_VISIBLE_DEVICES"] = " "  # only use CPU for training
tf.set_random_seed(99)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--num-gpu-core", type=int, default=1, help="Number of GPU cores to use")
    parser.add_argument("--num-cpu-core", type=int, default=4, help="Number of CPU cores to use")
    parser.add_argument("--inter-op-parallelism-threads", type=int, default=4,
                        help="0 means the system picks an appropriate number.")
    parser.add_argument("--intra-op-parallelism-threads", type=int, default=4,
                        help="0 means the system picks an appropriate number.")
    parser.add_argument("--use-fp16", default=False, action='store_true', help="whether use float16 as default")
    args = parser.parse_args()

    print("Available devices:")
    print(device_lib.list_local_devices())

    config_dict = load_cfg_from_yaml('config/CIFAR10/R_50.yaml')
    config_dict = merge_cfg_from_args(config_dict, args)
    lr = config_dict["policy_lr"]
    intra_op_threads = config_dict['intra_op_parallelism_threads']
    inter_op_threads = config_dict['inter_op_parallelism_threads']
    use_fp16 = config_dict['use_fp16']
    cpu_num = config_dict["num_cpu_core"]
    gpu_num = config_dict["num_gpu_core"]
    MAX_EPOCH = config_dict['TRAIN']['MAX_EPOCH']
    device_id = -1

    config = tf.ConfigProto(
        # device_count limits the number of CPUs being used, not the number of cores or threads.
        device_count={'CPU': cpu_num, 'GPU': gpu_num},
        inter_op_parallelism_threads=cpu_num,  # parallel without each operation, i.e., reduce_sum
        intra_op_parallelism_threads=cpu_num,  # parallel between multiple operations
        log_device_placement=True,  # log the GPU or CPU device that is assigned to an operation
        allow_soft_placement=True,  # use soft constraints for the device placement
    )
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:
        # input for resnet
        tf_x = tf.placeholder(dtype=data_type(use_fp16), shape=[None, 32, 32, 3], name='tf_x')
        tf_y = tf.placeholder(dtype=data_type(use_fp16), shape=[None, 10], name='tf_y')
        # output of resnet
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            with tf.device(next_device(device_id, config_dict, use_cpu=False)):
                resnet_out, end_points = resnet_v2.resnet_v2_50(tf_x, num_classes=10, is_training=False)
        # loss
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=resnet_out, labels=tf_y))
        # tf_loss_summary = tf.summary.scalar('loss', cross_entropy_loss)
        # optimizer
        tf_lr = tf.placeholder(dtype=data_type(use_fp16), shape=None, name='learning_rate')  # more flexible learning rate
        optimizer = tf.train.AdamOptimizer(tf_lr)
        grads_and_vars = optimizer.compute_gradients(cross_entropy_loss)
        train_step = optimizer.minimize(cross_entropy_loss)

        correct_prediction = tf.equal(tf.argmax(resnet_out, 1), tf.argmax(tf_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # tf_accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # Name scope allows you to group various summaries together
        # Summaries having the same name_scope will be displayed on the same row
        with tf.name_scope('performance'):
            # Summaries need to be displayed
            # Whenever you need to record the loss, feed the mean loss to this placeholder
            tf_loss_ph = tf.placeholder(dtype=data_type(use_fp16), shape=None, name='loss_summary')
            # Create a scalar summary object for the loss so it can be displayed
            tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

            # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
            tf_accuracy_ph = tf.placeholder(dtype=data_type(use_fp16), shape=None, name='accuracy_summary')
            # Create a scalar summary object for the accuracy so it can be displayed
            tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

        # Gradient norm summary
        # tf_gradnorm_summary: this calculates the l2 norm of the gradients of the last layer
        # of your neural network. Gradient norm is a good indicator of whether the weights of
        # the neural network are being properly updated. A too small gradient norm can indicate
        # vanishing gradient or a too large gradient can imply exploding gradient phenomenon.
        t_layer = len(grads_and_vars) - 2  # index of the last layer
        for i_layer, (g, v) in enumerate(grads_and_vars):
            if i_layer == t_layer:
                with tf.name_scope('gradients_norm'):
                    tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g ** 2))
                    tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)
                    break

        # A summary for each weight in each layer of ResNet50
        all_summaries = []
        for weight_name in end_points:
            try:
                name_scope = weight_name.split("bottleneck_v2/")[0] + weight_name.split("bottleneck_v2/")[1]
            except:
                name_scope = weight_name.split("bottleneck_v2/")[0]
            with tf.name_scope(name_scope):
                weight = end_points[weight_name]
                # Create a scalar summary object for the loss so it can be displayed
                tf_w_hist = tf.summary.histogram('weights_hist', tf.reshape(weight, [-1]))
                all_summaries.append([tf_w_hist])
        # Merge all parameter histogram summaries together
        tf_weight_summaries = tf.summary.merge(all_summaries)

        # Merge all summaries together
        # the following two statements are equal
        # [1] merge_op = tf.summary.merge_all()
        # [2] merge_op = tf.summary.merge([tf_loss_summary, tf_accuracy_summary, tf_gradnorm_summary, ...])
        merge_op = tf.summary.merge([tf_loss_summary, tf_accuracy_summary])
        # separate tensorboard, i.e., one log file, one folder.
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)

        # get cifar10 dataset
        cifar10_data = Cifar10('dataset/CIFAR10/', config=config_dict)
        test_images, test_labels = cifar10_data.test_data()

        # training
        start_time = time.time()
        batch_counter = 0
        while cifar10_data.epochs_completed < MAX_EPOCH:
            batch_xs, batch_ys = cifar10_data.next_train_batch()
            batch_counter += 1
            sess.run(train_step, feed_dict={tf_x: batch_xs, tf_y: batch_ys, tf_lr: lr})

            # calculate the train_accuracy for one batch
            if batch_counter % 100 == 0:
                train_accuracy, loss = sess.run(
                    [accuracy, cross_entropy_loss], feed_dict={tf_x: batch_xs, tf_y: batch_ys, tf_lr: lr})
                train_summary = sess.run(
                    merge_op, feed_dict={tf_loss_ph: loss, tf_accuracy_ph: train_accuracy})
                train_writer.add_summary(train_summary, batch_counter)
                print("----- epoch {} batch {} training accuracy {} loss {}".
                      format(cifar10_data.epochs_completed, batch_counter, train_accuracy, loss))

                end_time = time.time()
                print("time: {}".format(end_time - start_time))
                start_time = end_time

            # calculate the gradient norm summary and weight histogram
            if batch_counter % 100 == 0:
                gn_summary, wb_summary = sess.run(
                    [tf_gradnorm_summary, tf_weight_summaries], feed_dict={tf_x: batch_xs, tf_y: batch_ys, tf_lr: lr})
                train_writer.add_summary(gn_summary, batch_counter)
                train_writer.add_summary(wb_summary, batch_counter)

            # calculate the test_accuracy
            if batch_counter % 1000 == 0:
                # Test_accuracy
                test_accuracy, test_loss = sess.run(
                    [accuracy, cross_entropy_loss], feed_dict={tf_x: test_images, tf_y: test_labels, tf_lr: lr})
                test_summary = sess.run(
                    merge_op, feed_dict={tf_loss_ph: test_loss, tf_accuracy_ph: test_accuracy})
                test_writer.add_summary(test_summary, batch_counter)
                print("----- test accuracy {} test loss {}".format(test_accuracy, test_loss))

        # Overall test accuracy
        overall_accuracy = accuracy.eval(feed_dict={tf_x: test_images, tf_y: test_labels, tf_lr: lr})
        print("\nOverall test accuracy {}".format(overall_accuracy))

        save_cfg_to_yaml(config_dict, 'logs/current_config.yaml')


if __name__ == '__main__':
    main()
