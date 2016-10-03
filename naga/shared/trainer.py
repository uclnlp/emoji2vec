import tensorflow as tf


class Trainer(object):
    """
    Object representing a TensorFlow trainer.
    """

    def __init__(self, optimizer, max_epochs, hooks=[]):
        self.loss = None
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.hooks = hooks

    def __call__(self, batcher, placeholders, loss, model=None, session=None):
        self.loss = loss
        minimization_op = self.optimizer.minimize(loss)
        close_session_after_training = False
        if session is None:
            session = tf.Session()
            close_session_after_training = True  # no session existed before, we provide a temporary session

        init = tf.initialize_all_variables()
        session.run(init)
        epoch = 1
        iteration = 1

        while epoch < self.max_epochs:
            for values in batcher:
                feed_dict = {}
                for i in range(0, len(placeholders)):
                    feed_dict[placeholders[i]] = values[i]
                _, current_loss = session.run([minimization_op, loss], feed_dict=feed_dict)
                current_loss = sum(current_loss)
                for hook in self.hooks:
                    hook(session, epoch, iteration, model, current_loss)
                iteration += 1

            # calling post-epoch hooks
            for hook in self.hooks:
                hook(session, epoch, 0, model, 0)
            epoch += 1

        if close_session_after_training:
            session.close()
