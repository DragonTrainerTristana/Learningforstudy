# Code is mainly from https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf
from builtins import range
if tf.__version__ == '1.0.0':
    from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
else:
    # Tensorflow 1.4
    from tensorflow.python.ops import rnn_cell_impl as core_rnn_cell_impl


# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError(
            "Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.compat.v1.variable_scope(scope or "SimpleLinear"):
        matrix = tf.compat.v1.get_variable(
            "Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.compat.v1.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.compat.v1.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(
                linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=1.0, wgan_reg_lambda=1.0, grad_clip=1.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(
            tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        self.d_count = 0

        with tf.compat.v1.variable_scope('discriminator'):

            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.compat.v1.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(
                    self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-{:s}".format(str(filter_size))):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filter]
                    W = tf.Variable(tf.compat.v1.truncated_normal(
                        filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(
                        0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(
                    self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(
                    self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.compat.v1.truncated_normal(
                    [num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(
                    self.scores, 1, name="predictions")
                self.l2_loss = l2_reg_lambda * l2_loss
                self.s_l2_loss = tf.summary.scalar(
                    "l2_loss", self.l2_loss)
                self.crossentropy_loss = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))
                self.s_crossentropy_loss = tf.summary.scalar(
                    "crossentropy_loss", self.crossentropy_loss)

            # =Wasserstein loss
            with tf.name_scope("loss"):
                negs = tf.cast(self.input_y[:, 0], tf.int32)
                pos = tf.cast(self.input_y[:, 1], tf.int32)

                parts = tf.dynamic_partition(self.scores, pos, 2)
                scores_neg = parts[0]
                scores_pos = parts[1]


                #xy_neg = tf.boolean_mask(scores_t, negs)
                #xy_pos = tf.boolean_mask(scores_t, negs)

                #xy_pos = tf.boolean_mask(scores_t, pos)
                # self.s_xy_neg_shape = tf.summary.histogram("xy_neg_shape",
                #                                            xy_neg / tf.cast(tf.shape(xy_neg)[0], tf.float32))
                # self.s_xy_pos_shape = tf.summary.histogram("xy_pos_shape",
                #                                            xy_pos / tf.cast(tf.shape(xy_pos)[0], tf.float32))

                wgan_loss = tf.abs(
                    tf.reduce_sum(scores_neg) / tf.cast(tf.shape(scores_neg)[0], tf.float32) -
                    tf.reduce_sum(scores_pos) / tf.cast(tf.shape(scores_pos)[0], tf.float32))

                self.wgan_loss = wgan_reg_lambda * wgan_loss
                self.loss = self.l2_loss + self.wgan_loss

                self.s_loss = tf.summary.scalar("total_loss", self.loss)
                self.s_wgan_loss = tf.summary.scalar(
                    "wgan_loss", self.wgan_loss)

        with tf.name_scope("train"):
            self.params = [param for param in tf.compat.v1.trainable_variables(
            ) if 'discriminator' in param.name]

            self.optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
            # k Lipschitz constraint, copied from FusionGAN
            # grad_pen_gav = d_optimizer.compute_gradients(self.loss, [self.h_highway])
            # grad_pen = tf.reduce_mean([grad for grad, var in grad_pen_gav])
            # self.s_grad_pen = tf.summary.scalar("grad_pen", grad_pen)
            grad_pen = 0

            grads_and_vars = self.optimizer.compute_gradients(
                self.loss + grad_pen, self.params, aggregation_method=2)
            capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var)
                          for grad, var in grads_and_vars]

            # self.grads_and_vars_summ = []
            # for grad, var in grads_and_vars:
            #     x = tf.summary.histogram(var, grad)
            #     self.grads_and_vars_summ.append(x)
            self.train_op = self.optimizer.apply_gradients(capped_gvs)

        return

    def generate_summary(self, sess, x_batch, y_batch, dis_dropout_keep_prob):
        feed = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: dis_dropout_keep_prob
        }
        _summ = sess.run(
            tf.summary.merge(
                [self.s_loss,
                 self.s_crossentropy_loss,
                 self.s_l2_loss,
                 self.s_wgan_loss
                 ]
            ),
            feed)
        cur_d_count = self.d_count
        #self.d_count += 1
        return cur_d_count, _summ

    def train(self, sess, x_batch, y_batch, dis_dropout_keep_prob):
        feed = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: dis_dropout_keep_prob
        }

        return sess.run([self.train_op, self.loss,
                         self.crossentropy_loss,
                         self.l2_loss,
                         self.s_wgan_loss], feed)

    def get_score(self, sess, x_batch, dis_dropout_keep_prob):
        feed = {
            self.input_x: x_batch,
            self.dropout_keep_prob: dis_dropout_keep_prob
        }
        return sess.run([self.scores], feed)
