import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import classification_report


LOSS_TRACE_TAG = "Loss"
SPEED_TRACE_TAG = "Speed"
ACCURACY_TRACE_TAG = "Accuracy"

# todo: hooks should also have prefixes so that one can use the same hook with different parameters
class Hook(object):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, sess, epoch, iteration, model, loss):
        raise NotImplementedError


class TraceHook(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer

    def __call__(self, sess, epoch, iteration, model, loss):
        raise NotImplementedError

    def update_summary(self, sess, current_step, title, value):
        cur_summary = tf.summary.scalar(title, value)
        merged_summary_op = tf.summary.merge([cur_summary])  # if you are using some summaries, merge them
        summary_str = sess.run(merged_summary_op)
        self.summary_writer.add_summary(summary_str, current_step)


class LossHook(TraceHook):
    def __init__(self, summary_writer, iteration_interval, batch_size):
        super().__init__(summary_writer)
        self.iteration_interval = iteration_interval
        self.acc_loss = 0
        self.batch_size = batch_size

    def __call__(self, sess, epoch, iteration, model, loss):
        self.acc_loss += loss / self.batch_size
        if not iteration == 0 and iteration % self.iteration_interval == 0:
            loss = self.acc_loss / self.iteration_interval
            print("Epoch " + str(epoch) +
                  "\tIter " + str(iteration) +
                  "\tLoss " + str(loss))
            self.update_summary(sess, iteration, LOSS_TRACE_TAG, loss)
            self.acc_loss = 0


class SpeedHook(TraceHook):
    def __init__(self, summary_writer, iteration_interval, batch_size):
        super().__init__(summary_writer)
        self.iteration_interval = iteration_interval
        self.batch_size = batch_size
        self.t0 = time.time()
        self.num_examples = iteration_interval * batch_size

    def __call__(self, sess, epoch, iteration, model, loss):
        if not iteration == 0 and iteration % self.iteration_interval == 0:
            diff = time.time() - self.t0
            speed = int(self.num_examples / diff)
            print("Epoch " + str(epoch) +
                  "\tIter " + str(iteration) +
                  "\tExamples/s " + str(speed))
            self.update_summary(sess, iteration, SPEED_TRACE_TAG, float(speed))
            self.t0 = time.time()
