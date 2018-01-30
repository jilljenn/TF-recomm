import tensorflow as tf
import numpy as np
from svd_train_val import NB_CLASSES


def inference_svd(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    with tf.device("/cpu:0"):
        bias_global = tf.get_variable("bias_global", shape=[])
        bias = tf.get_variable("bias", shape=[user_num + item_num], initializer=tf.truncated_normal_initializer(stddev=10))
        bias_user = tf.nn.embedding_lookup(bias, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(bias, user_num + item_batch, name="bias_item")

        all_log_thresholds = tf.get_variable("log_thresholds", shape=[item_num, NB_CLASSES - 1],
                                 initializer=tf.truncated_normal_initializer(stddev=10))
        log_thresholds_item = tf.nn.embedding_lookup(all_log_thresholds, item_batch, name="thre_item")

        features = tf.get_variable("features", shape=[user_num + item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(features, user_batch, name="feat_user")
        embd_item = tf.nn.embedding_lookup(features, user_num + item_batch, name="feat_item")
    with tf.device(device):
        logits = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        #logits = tf.add(logits, bias_global)
        logits = tf.add(logits, bias_user)
        #logits = tf.add(infer, bias_item, name="svd_inference")
        print(logits[:, None])
        cumulative_op = tf.constant(np.tri(NB_CLASSES - 1).T, dtype=tf.float32)
        thresholds_item = tf.matmul(tf.abs(log_thresholds_item), cumulative_op) - bias_item[:, None]
        cdf = tf.sigmoid(logits[:, None] - thresholds_item)

        # Ordinal regression
        pdf2cdf_A = tf.constant(np.fromfunction(lambda i, j: (j == i + 1) - 1. * (j == i), (NB_CLASSES - 1, NB_CLASSES), dtype=float), dtype=tf.float32)
        pdf2cdf_b = tf.constant(np.fromfunction(lambda i, j: 1. * (j == 0), (1, NB_CLASSES), dtype=float), dtype=tf.float32)
        print(cdf)
        print(pdf2cdf_A)
        pdf = tf.matmul(cdf, pdf2cdf_A) + pdf2cdf_b
        infer = tf.argmax(pdf, axis=1)

        l2_user = tf.nn.l2_loss(embd_user)
        l1_user = tf.reduce_mean(tf.abs(embd_user))
        l2_item = tf.nn.l2_loss(embd_item)
        l1_item = tf.reduce_mean(tf.abs(embd_item))
        regularizer = tf.add(l1_item, l1_item, name="svd_regularizer")
    return infer, logits, regularizer, thresholds_item


def logloss(x):
    return tf.log(1 + tf.exp(-x))


def immediate_thresholds(labels, logits, thresholds):
    signs = tf.one_hot(labels, NB_CLASSES - 1) - tf.one_hot(labels - 1, NB_CLASSES - 1)
    delta = logits[:, None] - thresholds
    return tf.reduce_sum(logloss(signs * delta), axis=1)


def all_thresholds(labels, logits, thresholds):
    signs = tf.sign(2 * (tf.range(NB_CLASSES - 1, dtype=tf.float32) - tf.cast(labels[:, None], tf.float32)) + 1)
    delta = logits[:, None] - thresholds
    return tf.reduce_sum(logloss(signs * delta), axis=1)


def optimization(infer, logits, regularizer, rate_batch, thresholds, learning_rate, reg, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        #cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        #cost_nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch, logits=infer)
        #cost_at = immediate_thresholds(labels=rate_batch, logits=logits, thresholds=thresholds)
        cost_at = all_thresholds(labels=rate_batch, logits=logits, thresholds=thresholds)
        #auc, update_op = tf.metrics.auc(rate_batch, tf.sigmoid(infer))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        #cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        #cost = tf.add(cost_nll, tf.multiply(regularizer, penalty))
        cost = tf.add(cost_at, tf.multiply(regularizer, penalty))
        #train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
        #train_item_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
    #return cost_l2, train_op  # If not discrete
    #return cost_nll, auc, update_op, train_op
    return cost_at, train_op
