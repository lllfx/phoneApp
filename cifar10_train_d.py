

from __future__ import print_function
import tensorflow as tf
import sys
import time
import cifar10
from datetime import datetime

# cluster specification
parameter_servers = ["10.63.73.120:2223"]
workers = ["10.63.73.177:2223",
           "10.63.73.122:2223"]
cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)
# config
batch_size = 128
learning_rate = 0.001
max_steps = 1000

# Get images and labels for CIFAR-10.
images, labels = cifar10.distorted_inputs()

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # Calculate loss.
        loss = cifar10.loss(logits, labels)

        # specify optimizer
        with tf.name_scope('train'):
            # optimizer is an "operation" which we can execute in a session
            grad_op = tf.train.GradientDescentOptimizer(learning_rate)
            '''
            rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                                                                    replicas_to_aggregate=len(workers),
                                                                                     replica_id=FLAGS.task_index,                                                       total_num_replicas=len(workers),
                                                                                     use_locking=True
                                                                                  )
             train_op = rep_op.minimize(cross_entropy, global_step=global_step)
             '''
            train_op = grad_op.minimize(loss, global_step=global_step)

        '''
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
        '''

        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             global_step=global_step,
                             init_op=init_op)

    begin_time = time.time()

    with sv.prepare_or_wait_for_session(server.target) as sess:
        '''
        # is chief
        if FLAGS.task_index == 0:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_token_op)
        '''
        # perform training cycles
        start_time = time.time()
        for step in range(FLAGS.max_steps):
            # perform the operations we defined earlier on batch
            _, cost, step = sess.run([train_op, loss, global_step])
            print("Step: %d," % (step + 1)," Cost: %.4f," % cost)

        print("Total Time: %3.2fs" % float(time.time() - begin_time))
        print("Final Cost: %.4f" % loss)

    sv.stop()
    print("done")