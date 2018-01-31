from config import *
from collections import deque
import tensorflow as tf
import numpy as np
import os.path
import dataio
from cats import Random
import time
import ops

LEARNING_RATE = 1
EPOCH_MAX = 84
BUDGET = 10

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
rate_batch = tf.placeholder(tf.int32, shape=[None])

infer, logits_cdf, pdf, regularizer, user_bias, user_features = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM, device=DEVICE)
global_step = tf.train.get_or_create_global_step()
# Attention: only var_list = embd_user, bias_user
cost, train_op = ops.optimization(infer, logits_cdf, logits_pdf, regularizer, rate_batch, learning_rate=LEARNING_RATE, reg=LAMBDA_REG, device=DEVICE, var_list=[user_bias, user_features])

_, _, df_test = dataio.get_data()

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, os.path.join(BASE_DIR, "fm.ckpt"))

    all_user_features = sess.run(user_features, feed_dict={user_batch: range(USER_NUM)})
    all_user_features_norms = np.diag(all_user_features.dot(all_user_features.T))
    all_user_bias = sess.run(user_bias, feed_dict={user_batch: range(USER_NUM)})
    print('all_features', all_user_features.min(), 'to', all_user_features.max())
    print('all_features_norms', all_user_features_norms.min(), 'to', all_user_features_norms.max())
    print('all_bias', all_user_bias.min(), 'to', all_user_bias.max())
    start = time.time()

    cat = Random(ITEM_NUM)
    for b in range(BUDGET):

        

        for i in range(EPOCH_MAX):
            test_users = df_test['user']
            test_items = df_test['item']
            test_rates = df_test['rate']
            train_users = test_users[:10]
            train_items = test_items[:10]
            train_rates = test_rates[:10]

            _, train_logits_cdf, train_infer = sess.run(
                    [train_op, logits_cdf, infer], feed_dict={
                        user_batch: train_users, item_batch: train_items, rate_batch: train_rates})

            train_cost = deque()
            train_acc = deque()
            train_obo = deque()
            cost_batch = sess.run(cost, feed_dict={rate_batch: train_rates, item_batch: train_items,
                                                   user_batch: train_users, logits_cdf: train_logits_cdf})
            train_cost.append(cost_batch)
            train_acc.append(train_infer == train_rates)
            train_obo.append(abs(train_infer - train_rates) <= 1)
            train_macc = np.mean(train_acc)
            train_mobo = np.mean(train_obo)
            train_mcost = np.mean(train_cost)

            test_logits_cdf, test_infer = sess.run(
                    [logits_cdf, infer], feed_dict={
                        user_batch: test_users, item_batch: test_items, rate_batch: test_rates})

            test_cost = deque()
            test_acc = deque()
            test_obo = deque()
            cost_batch = sess.run(cost, feed_dict={rate_batch: test_rates, item_batch: test_items,
                                                   user_batch: test_users, logits_cdf: test_logits_cdf})
            test_cost.append(cost_batch)
            test_acc.append(test_infer == test_rates)
            test_obo.append(abs(test_infer - test_rates) <= 1)
            test_macc = np.mean(test_acc)
            test_mobo = np.mean(test_obo)
            test_mcost = np.mean(test_cost)

            if DISCRETE:
                if NB_CLASSES > 2:
                    end = time.time()
                    print("{:3d} TRAIN(size={:d}, macc={:f}, mobo={:f}, mcost={:f}) TEST(size={:d}, macc={:f}, mobo={:f}, mcost={:f}) {:f}(s)".format(
                        i,
                        len(train_users),
                        train_macc,
                        train_mobo,
                        train_mcost,
                        len(test_users),
                        test_macc,
                        test_mobo,
                        test_mcost,
                        end - start))
            start = end
