"""
Github issue: https://github.com/tensorflow/tensorflow/issues/22619
"""
from __future__ import print_function

import argparse
import time

import numpy as np
import tensorflow.compat.v1 as tf

parser = argparse.ArgumentParser()
# nargs='+': all command-line args present are gathered into a list and at least
# one command-line argument present
# 'core_counts': argument without '--' must be provided when being called.
parser.add_argument('core_counts', nargs='+', type=int)
parser.add_argument('--use-devs', action='store_true')
parser.add_argument('--use-inter', action='store_true')
parser.add_argument('--use-intra', action='store_true')
parser.add_argument('--no-const-fold', action='store_false', dest='const_fold')
args = parser.parse_args()

# print(args.core_counts)  # [2]
print(args)

for n_cpus in args.core_counts:
    n_devs = n_cpus if args.use_devs else 1
    n_inter = n_cpus if args.use_inter else 1
    n_intra = n_cpus if args.use_intra else 1

    config = tf.ConfigProto(
        device_count={"CPU": n_devs},
        inter_op_parallelism_threads=n_inter,
        intra_op_parallelism_threads=n_intra,
    )
    with tf.Session(config=config) as sess:

        print('Running on %s CPU devices with %s inter- and %s intra-parallelism' % (n_devs, n_inter, n_intra))

        size = 7000

        ops = []
        feed = {}
        for i in range(n_cpus):
            d = "/cpu:%s" % (i % n_devs)
            print('  Assigning matmul to {} -- {}'.format(d, n_cpus))
            with tf.device(d):
                if args.const_fold:
                    A = tf.ones([size, size], name=("A%s" % i))
                    B = tf.ones([size, size], name=("B%s" % i))
                else:
                    A_name = "A%s" % i
                    B_name = "B%s" % i
                    A = tf.placeholder(tf.float32, shape=[size, size], name=A_name)
                    B = tf.placeholder(tf.float32, shape=[size, size], name=B_name)
                    feed["%s:0" % A_name] = np.random.rand(size, size)
                    feed["%s:0" % B_name] = np.random.rand(size, size)
                x = tf.matmul(A, B)
                ops.append(x)

        start_perf_counter = time.perf_counter()
        start_time = time.time()
        start_clock = time.clock()
        sess.run(ops, feed_dict=feed)
        stop_perf_counter = time.perf_counter()
        stop_time = time.time()
        stop_clock = time.clock()

        print('  Duration (via time.perf_counter()): %f (%f - %f)' %
              (stop_perf_counter - start_perf_counter, stop_perf_counter, start_perf_counter))
        print('  Time (via time.time()): %f (%f - %f)' % (stop_time - start_time, stop_time, start_time))
        print('  Clock (via time.clock()): %f (%f - %f)' % (stop_clock - start_clock, stop_clock, start_clock))

        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # sess.run([x, y, z], options=run_options, run_metadata=run_metadata)

        # for device in run_metadata.step_stats.dev_stats:
        #     device_name = device.device
        #     print(device.device)
        #     for node in device.node_stats:
        #         print("   ", node.node_name)

        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('timeline_01.json', 'w') as f:
        #     f.write(chrome_trace)
