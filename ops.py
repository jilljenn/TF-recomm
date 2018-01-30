import tensorflow as tf
import numpy as np
from svd_train_val import NB_CLASSES


def inference_svd(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    with tf.device("/cpu:0"):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num + item_num])
        thresholds = tf.get_variable("thresholds", shape=[item_num, NB_CLASSES])
        w_user = tf.get_variable("embd_user", shape=[user_num + item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        #w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_user, user_num + item_batch, name="bias_item")
        thresholds_item = tf.nn.embedding_lookup(thresholds, item_batch, name="thr_item")
        #w_item = tf.get_variable("embd_item", shape=[item_num, dim],
        #                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_user, user_num + item_batch, name="embedding_item")
    with tf.device(device):
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        #infer = tf.add(infer, bias_item, name="svd_inference")
        l2_user = tf.nn.l2_loss(embd_user)
        l1_user = tf.reduce_mean(tf.abs(embd_user))
        l2_item = tf.nn.l2_loss(embd_item)
        l1_item = tf.reduce_mean(tf.abs(embd_item))
        regularizer = tf.add(l1_user, l1_item, name="svd_regularizer")
    return infer, regularizer, thresholds_item


def logloss(x):
    return tf.log(1 + tf.exp(-x))


def all_thresholds(labels, logits, thresholds):
    signs = tf.sign(2 * (tf.range(NB_CLASSES, dtype=tf.float32) - labels[:, None]) + 1)
    delta = logits[:, None] - thresholds
    return tf.reduce_sum(logloss(signs * delta), axis=1)


def optimization(infer, regularizer, rate_batch, thresholds, learning_rate, reg, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        #cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        #cost_nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch, logits=infer)
        cost_at = all_thresholds(labels=rate_batch, logits=infer, thresholds=thresholds)
        auc, update_op = tf.metrics.auc(rate_batch, tf.sigmoid(infer))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        #cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        #cost = tf.add(cost_nll, tf.multiply(regularizer, penalty))
        cost = tf.add(cost_at, tf.multiply(regularizer, penalty))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    #return cost_l2, train_op  # If not discrete
    return cost_at, auc, update_op, train_op
