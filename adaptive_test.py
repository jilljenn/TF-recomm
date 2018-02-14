from config import *
from collections import deque, Counter
import tensorflow as tf
import numpy as np
import os.path
import dataio
from cats import Random, Fisher, Popular, Next
import time
import ops

LEARNING_RATE = 5 * 1e-3
EPOCH_MAX = 300
BUDGET = 10
LAMBDA_REG = 0
ASK_EVERYTHING = False
LOG_STEP = 300
COMPUTE_TEST = False

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
# count_work = df_train.groupby('item').size().sort_index()
# work_ids = count_work.index
# work_count = count_work.values
# popularity = np.zeros(len(work_ids))
# popularity[work_ids] = work_count

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

    unique_test_users = df_test['user'].unique()

    for this_user in unique_test_users[:3]:  # FIXME
        print()
        print('Welcome', this_user)

        df_user = df_test.query('user == @this_user')

        test_users = df_user['user']
        test_items = df_user['item']
        test_rates = df_user['outcome']
        test_wins = df_user['wins']
        test_fails = df_user['fails']

        train_users = []
        train_items = []
        train_rates = []
        train_wins = []
        train_fails = []

        if ASK_EVERYTHING:
            train_users = test_users
            train_items = test_items
            train_rates = test_rates
            train_wins = test_wins
            train_fails = test_fails

        cat = Next(test_items)
        print('First items are', list(test_items)[:5])

        for b in range(BUDGET):
            # train_logits_cdf, train_infer, train_logits_pdf, train_item_bias, train_item_features, train_user_bias, train_user_features, train_thresholds = sess.run(
            #     [logits_cdf, infer, logits_pdf, item_bias, item_features, user_bias, user_features, thresholds], feed_dict={
            #         user_batch: [this_user] * ITEM_NUM, item_batch: range(ITEM_NUM)})
            # train_cdf = ops.sigmoid(train_logits_cdf)
            # train_pdf = ops.sigmoid(train_logits_pdf)

            # cat.update_probas(train_cdf, train_pdf, train_item_bias, train_item_features)
            item_to_ask = cat.next_item()

            if not ASK_EVERYTHING:
                train_users.append(this_user)
                train_items.append(item_to_ask)
                # print('trained on', train_items)
                # print(item_to_ask in test_items)
                this_entry = df_user.query('item == @item_to_ask')
                train_rates.append(list(this_entry['outcome'])[0])
                train_wins.append(list(this_entry['wins'])[0])
                train_fails.append(list(this_entry['fails'])[0])
                # print(train_users, train_items, train_rates)

            item_logit = sess.run(logits, feed_dict={
                user_batch: [this_user], item_batch: [item_to_ask], wins_batch: train_wins[-1:], fails_batch: train_fails[-1:]})
            item_proba = ops.sigmoid(item_logit)

            print(this_entry.index, 'Asking', item_to_ask, 'predicted', item_proba, 'actually', train_rates[-1])

            for i in range(1, EPOCH_MAX + 1):

                # if i % 50 == 0:
                #     print('training that does not change parameters?', train_logits_cdf[0])
                #     print('user bias', train_user_bias[:5])
                #     print('item bias', train_item_bias[0])
                #     print('item thresholds', train_thresholds[0])
                #     print('user features', train_user_features[:5])

                _, train_logits_cdf, train_infer, train_user_bias, train_item_bias, train_user_features, train_thresholds = sess.run(
                        [train_op, logits_cdf, infer, user_bias, item_bias, user_features, thresholds], feed_dict={
                            user_batch: train_users, item_batch: train_items, rate_batch: train_rates, wins_batch: train_wins, fails_batch: train_fails})

                # if i % 50 == 0:
                #     print('training that does not change parameters?', train_logits_cdf[0])
                #     print('user bias', train_user_bias[:5])
                #     print('item bias', train_item_bias[0])
                #     print('item thresholds', train_thresholds[0])
                #     print('user features', train_user_features[:5])

                if i % LOG_STEP == 0:

                    train_cost = deque()
                    train_acc = deque()
                    train_obo = deque()
                    train_se = deque()
                    cost_batch = sess.run(cost, feed_dict={rate_batch: train_rates, item_batch: train_items,
                                                           user_batch: train_users, wins_batch: train_wins, fails_batch: train_fails, logits_cdf: train_logits_cdf})
                    train_cost.append(cost_batch)
                    train_acc.append(train_infer == train_rates)
                    train_obo.append(abs(train_infer - train_rates) <= 1)
                    # print(Counter(abs(train_infer - train_rates)))
                    train_se.append(np.power(train_infer - train_rates, 2))
                    train_macc = np.mean(train_acc)
                    train_mobo = np.mean(train_obo)
                    train_mcost = np.mean(train_cost)
                    train_rmse = np.sqrt(np.mean(train_se))

                if COMPUTE_TEST and i % LOG_STEP == 0:
                    test_logits, test_logits_cdf, test_infer, all_user_bias, all_user_features = sess.run(
                            [logits, logits_cdf, infer, user_bias, user_features], feed_dict={
                                user_batch: test_users, item_batch: test_items, rate_batch: test_rates})

                    test_cost = deque()
                    test_acc = deque()
                    test_obo = deque()
                    test_se = deque()
                    cost_batch = sess.run(cost, feed_dict={rate_batch: test_rates, item_batch: test_items,
                                                           user_batch: test_users, logits_cdf: test_logits_cdf})
                    test_cost.append(cost_batch)
                    test_acc.append(test_infer == test_rates)
                    test_obo.append(abs(test_infer - test_rates) <= 1)
                    test_se.append(np.power(test_infer - test_rates, 2))
                    test_macc = np.mean(test_acc)
                    test_mobo = np.mean(test_obo)
                    test_mcost = np.mean(test_cost)
                    test_rmse = np.sqrt(np.mean(test_se))

                    if DISCRETE:
                        if NB_CLASSES > 2:
                            end = time.time()
                            print("{:d} {:d} {:3d} TRAIN(size={:d}, macc={:f}, mobo={:f}, rmse={:f}, mcost={:f}, bias={:f}, features={:f}) TEST(size={:d}, macc={:f}, mobo={:f}, rmse={:f}, mcost={:f}) {:f}(s)".format(
                                this_user,
                                i,
                                b,  # Budget
                                len(train_users),
                                train_macc,
                                train_mobo,
                                train_rmse,
                                train_mcost,
                                all_user_bias[this_user],
                                all_user_features[this_user].mean(),
                                len(test_users),
                                test_macc,
                                test_mobo,
                                test_rmse,
                                test_mcost,
                                # abs(all_user_bias[this_user] - 2.251978),
                                # abs(all_user_features[this_user].mean() + 0.720547),
                                end - start))
                        start = end

# with open('feed.txt', 'w') as f:
#     df_user['pred'] = test_logits
#     f.write(str(list(df_user.sort_values('pred', ascending=False)['item'])))
