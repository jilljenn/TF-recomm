from config import *
from collections import deque, Counter, defaultdict
import tensorflow as tf
import numpy as np
import os.path
import dataio
from cats import Random, Fisher, Popular, Next
import time
import ops


LEARNING_RATE = 5 * 1e-3
EPOCH_MAX = 300
LAMBDA_REG = 0
LOG_STEP = 300


user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
rate_batch = tf.placeholder(tf.float32, shape=[None])
wins_batch = tf.placeholder(tf.float32, shape=[None], name="nb_wins")
fails_batch = tf.placeholder(tf.float32, shape=[None], name="nb_fails")

infer, logits, logits_cdf, logits_pdf, regularizer, user_bias, user_features, item_bias, item_features, thresholds = ops.inference_svd(user_batch, item_batch, wins_batch, fails_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM, device=DEVICE)
global_step = tf.train.get_or_create_global_step()
# Attention: only var_list = embd_user, bias_user
cost, auc, update_op, train_op = ops.optimization(infer, logits, logits_cdf, logits_pdf, regularizer, rate_batch, learning_rate=LEARNING_RATE, reg=LAMBDA_REG, device=DEVICE, var_list=[user_bias, user_features])

df_train, _, df_test = dataio.get_data()

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, os.path.join(BASE_DIR, "fm.ckpt"))

    all_user_features = sess.run(user_features, feed_dict={user_batch: range(USER_NUM)})
    all_user_features_norms = np.diag(all_user_features.dot(all_user_features.T))
    all_user_bias = sess.run(user_bias, feed_dict={user_batch: range(USER_NUM)})
    # print('all_features', all_user_features.min(), 'to', all_user_features.max())
    # print('all_features_norms', all_user_features_norms.min(), 'to', all_user_features_norms.max())
    # print('all_bias', all_user_bias.min(), 'to', all_user_bias.max())
    #print('item_features', all_user_bias.min(), 'to', all_user_bias.max())
    start = time.time()

    # unique_test_users = df_test['user'].unique()

    train_users = defaultdict(list)
    train_items = defaultdict(list)
    train_rates = defaultdict(list)
    train_wins = defaultdict(list)
    train_fails = defaultdict(list)

    for user_id, item_id, outcome, nb_wins, nb_fails in np.array(df_test):
        if user_id > 3374:
            break

        print()
        print('Welcome', user_id)

        train_users[user_id].append(user_id)
        train_items[user_id].append(item_id)
        train_rates[user_id].append(outcome)
        train_wins[user_id].append(nb_wins)
        train_fails[user_id].append(nb_fails)

        item_logit = sess.run(logits, feed_dict={
            user_batch: [user_id], item_batch: [item_id], wins_batch: [nb_wins], fails_batch: [nb_fails]})
        item_proba = ops.sigmoid(item_logit)

        print('Asking', item_id, 'predicted', item_proba, 'actually', outcome)

        for i in range(1, EPOCH_MAX + 1):

            _, train_logits_cdf, train_infer, train_user_bias, train_item_bias, train_user_features, train_thresholds = sess.run(
                    [train_op, logits_cdf, infer, user_bias, item_bias, user_features, thresholds], feed_dict={
                        user_batch: train_users[user_id], item_batch: train_items[user_id], rate_batch: train_rates[user_id], wins_batch: train_wins[user_id], fails_batch: train_fails[user_id]})

            if i % LOG_STEP == 0:

                train_cost = deque()
                train_acc = deque()
                train_obo = deque()
                train_se = deque()
                cost_batch = sess.run(cost, feed_dict={rate_batch: train_rates[user_id], item_batch: train_items[user_id],
                                                       user_batch: train_users[user_id], wins_batch: train_wins[user_id], fails_batch: train_fails[user_id], logits_cdf: train_logits_cdf})
                train_cost.append(cost_batch)
                train_acc.append(train_infer == train_rates[user_id])
                train_obo.append(abs(train_infer - train_rates[user_id]) <= 1)
                # print(Counter(abs(train_infer - train_rates)))
                train_se.append(np.power(train_infer - train_rates[user_id], 2))
                train_macc = np.mean(train_acc)
                train_mobo = np.mean(train_obo)
                train_mcost = np.mean(train_cost)
                train_rmse = np.sqrt(np.mean(train_se))

                end = time.time()
                print(end - start)
                print("{:d} {:d} TRAIN(size={:d}, macc={:f}, mobo={:f}, rmse={:f}, mcost={:f}) {:f}(s)".format(
                    int(user_id),
                    i,
                    len(train_users[user_id]),
                    train_macc,
                    train_mobo,
                    train_rmse,
                    train_mcost,
                    end - start))
