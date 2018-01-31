import tensorflow as tf
import numpy as np
from svd_train_val import NB_CLASSES


def inference_svd(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    with tf.device("/cpu:0"):
        bias_global = tf.get_variable("bias_global", shape=[])
        user_bias = tf.get_variable("user_bias", shape=[user_num],
            initializer=tf.truncated_normal_initializer(stddev=1))
        item_bias = tf.get_variable("item_bias", shape=[item_num],
            initializer=tf.truncated_normal_initializer(stddev=1))
        bias_users = tf.nn.embedding_lookup(user_bias, user_batch, name="bias_users")
        bias_items = tf.nn.embedding_lookup(item_bias, item_batch, name="bias_items")

        thresholds = tf.get_variable("thresholds", shape=[item_num, NB_CLASSES - 1],
            initializer=tf.truncated_normal_initializer(stddev=1))
        threshold_items = tf.nn.embedding_lookup(thresholds, item_batch, name="thre_items")

        user_features = tf.get_variable("user_features", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        item_features = tf.get_variable("item_features", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        feat_users = tf.nn.embedding_lookup(user_features, user_batch, name="feat_users")
        feat_items = tf.nn.embedding_lookup(item_features, item_batch, name="feat_items")
    with tf.device(device):
        logits = tf.reduce_sum(tf.multiply(feat_users, feat_items), 1)
        #logits = tf.add(logits, bias_global)
        logits = tf.add(logits, bias_users)
        logits = tf.add(logits, bias_items, name="svd_inference")

        cumulative_op = tf.constant(np.tri(NB_CLASSES - 1).T, dtype=tf.float32)
        pos_threshold_items = tf.matmul(tf.exp(threshold_items), cumulative_op) #- bias_items[:, None]
        logits_cdf = logits[:, None] - pos_threshold_items
        
        # Computing pdf for ordinal regression (needed to get the inferred label)
        cdf = tf.sigmoid(logits_cdf)
        pdf2cdf_A = tf.constant(np.fromfunction(lambda i, j: (j == i + 1) - 1. * (j == i), (NB_CLASSES - 1, NB_CLASSES), dtype=float), dtype=tf.float32)
        pdf2cdf_b = tf.constant(np.fromfunction(lambda i, j: 1. * (j == 0), (1, NB_CLASSES), dtype=float), dtype=tf.float32)
        pdf = tf.matmul(cdf, pdf2cdf_A) + pdf2cdf_b
        #logits_pdf = tf.log(pdf / (1 - pdf))
        #test = -logits_cdf + tf.abs(threshold_items)
        logits_pdf = tf.concat((-logits_cdf[:, 0][:, None], threshold_items - tf.log(tf.exp(logits_cdf) + tf.exp(-logits_cdf + tf.abs(threshold_items) + 2))), 1)
        infer = tf.argmax(pdf, axis=1)

        # Regularization
        l2_user = tf.nn.l2_loss(feat_users)
        l1_user = tf.reduce_mean(tf.abs(feat_users))
        l2_item = tf.nn.l2_loss(feat_items)
        l1_item = tf.reduce_mean(tf.abs(feat_items))
        regularizer = tf.add(l1_user, l2_item)
        l2_bias_user = tf.nn.l2_loss(bias_users)
        l2_bias_item = tf.nn.l2_loss(bias_items)
        #regularizer = tf.add(regularizer, l2_bias_user)
        regularizer = tf.add(regularizer, l2_bias_item, name="svd_regularizer")
    return infer, logits_cdf, pdf, regularizer, user_bias, user_features, item_bias


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def optimization(infer, logits_cdf, regularizer, rate_batch, learning_rate, reg, device="/cpu:0", var_list=None):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        labels_cdf = tf.nn.relu(tf.sign(tf.cast(rate_batch[:, None], tf.float32) + 0.5 - tf.range(1, NB_CLASSES, dtype=tf.float32)))
        labels_pdf = tf.one_hot(rate_batch, depth=NB_CLASSES)
        #cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        #cost_nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch, logits=infer)
        #cost_at = immediate_thresholds(labels=rate_batch, logits=logits, thresholds=thresholds)
        #cost_at = all_thresholds(labels=rate_batch, logits=logits, thresholds=thresholds)
        #n = tf.shape(rate_batch)[0]
        # labels_window = tf.gather(tf.concat((tf.constant(np.array([0.5])[None, :]), labels_cdf), axis=1), rate_batch - 1, axis=1)
        # labels_window = tf.ones([n, 1])
        # logits_cdf_plus = tf.concat((100 * tf.ones([n, 1]), logits_cdf), axis=1)
        # logits_window = tf.gather(logits_cdf_plus, rate_batch, axis=1)
        cost_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_cdf, logits=logits_cdf)
        #cost_ce = tf.nn.softmax_cross_entropy_with_logits(labels=labels_pdf, logits=logits_pdf)
        #auc, update_op = tf.metrics.auc(rate_batch, tf.sigmoid(infer))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="penalty")
        #cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        #cost = tf.add(cost_nll, tf.multiply(regularizer, penalty))
        cost = tf.add(cost_ce, tf.multiply(regularizer, penalty))
        #print(cost)
        
        if var_list is None:
            # train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
        else:
            # train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step, var_list=var_list)
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step, var_list=var_list)
        # else:
        #     train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
    #return cost_l2, train_op  # If not discrete
    #return cost_nll, auc, update_op, train_op
    return cost_ce, train_op
