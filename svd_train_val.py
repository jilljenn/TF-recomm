import time
from collections import deque, Counter

import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2
from sklearn.metrics import roc_auc_score

from config import *
import dataio
import ops
import os.path

np.random.seed(13575)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def print_local_vars():
    print([(str(i.name), i.eval().sum()) for i in tf.local_variables()])

def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def svd(train, test):
    nb_batches = len(train) // BATCH_SIZE

    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["item"],
                                         train["rate"]],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.OneEpochIterator([test["user"],
                                         test["item"],
                                         test["rate"]],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.int32, shape=[None])

    infer, logits_cdf, logits_pdf, regularizer, user_bias, user_features = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM, device=DEVICE)
    global_step = tf.train.get_or_create_global_step()
    #cost_l2, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=LEARNING_RATE, reg=LAMBDA_REG, device=DEVICE)
    #cost_nll, auc, update_op, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=LEARNING_RATE, reg=LAMBDA_REG, device=DEVICE)
    cost, train_op = ops.optimization(infer, logits_cdf, logits_pdf, regularizer, rate_batch, learning_rate=LEARNING_RATE, reg=LAMBDA_REG, device=DEVICE)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        train_se = deque(maxlen=nb_batches)
        train_nll = deque(maxlen=nb_batches)
        train_cost = deque(maxlen=nb_batches)
        train_acc = deque(maxlen=nb_batches)
        train_obo = deque(maxlen=nb_batches)
        train_auc = deque(maxlen=nb_batches)
        start = time.time()
        for i in range(EPOCH_MAX * nb_batches):
            train_users, train_items, train_rates = next(iter_train)
            batch_size = len(train_rates)

            _, train_logits_cdf, train_logits_pdf, train_infer = sess.run(
                [train_op, logits_cdf, logits_pdf, infer], feed_dict={
                    user_batch: train_users, item_batch: train_items, rate_batch: train_rates})
            # This can be removed
            # all_thresholds = sess.run(thresholds, feed_dict={item_batch: range(ITEM_NUM)})

            # print(train_logits_cdf[42])
            # print(train_logits_pdf[42])
            # print(train_rates[42])

            if DISCRETE:
                if NB_CLASSES > 2:
                    cost_batch = sess.run(cost, feed_dict={rate_batch: train_rates, item_batch: train_items, user_batch: train_users,
                                                           logits_cdf: train_logits_cdf, logits_pdf: train_logits_pdf})
                    # print(train_users[42])
                    # print(train_items[42])
                    # print(train_logits_pdf[42])
                    # print(train_logits_cdf[42])
                    # print('thr', all_thresholds)
                    # print('infer', train_infer[42])
                    train_cost.append(cost_batch)
                    train_acc.append(train_infer == train_rates)
                    train_obo.append(abs(train_infer - train_rates) <= 1)
                else:
                    nll_batch = sess.run(cost_nll, feed_dict={rate_batch: train_rates, infer: train_infer})
                    proba_batch = sigmoid(train_pred_batch)
                    train_acc.append(np.round(proba_batch) == train_rates)
                    train_auc.append(roc_auc_score(train_rates, proba_batch))
                    train_nll.append(nll_batch)
            else:
                l2_batch = sess.run(cost_l2, feed_dict={rate_batch: train_rates, infer: train_infer})
                #print('est-ce', np.sum(np.power(train_rates - train_pred_batch, 2)))
                #print('que = ', l2_batch)
                #train_se.append(np.power(l2_batch, 2))
                train_se.append(np.power(train_rates - train_infer, 2))

            if i % nb_batches == 0:
                # Compute test error
                train_rmse = np.sqrt(np.mean(train_se))
                train_macc = np.mean(train_acc)
                train_mobo = np.mean(train_obo)
                train_mauc = np.mean(train_auc)
                train_mnll = np.mean(train_nll)
                train_mcost = np.mean(train_cost)
                test_se = []
                test_acc = []
                test_obo = []
                test_auc = []
                test_nll = []
                test_cost = []
                for test_users, test_items, test_rates in iter_test:
                    test_logits_cdf, test_logits_pdf, test_infer = sess.run(
                        [logits_cdf, logits_pdf, infer], feed_dict={user_batch: test_users, item_batch: test_items})
                    test_size = len(test_rates)

                    # print(test_logits_cdf[42], test_logits_pdf[42])
                    # print(test_infer[42], test_rates[42])

                    print(test_infer[42:47])
                    print(test_rates[42:47])
                    if DISCRETE:
                        if NB_CLASSES > 2:
                            cost_batch = sess.run(cost, feed_dict={rate_batch: test_rates, item_batch: test_items, user_batch: test_users,
                                                                   logits_cdf: test_logits_cdf, logits_pdf: test_logits_pdf})
                            #print(cost_batch)
                            test_cost.append(cost_batch)
                            test_acc.append(test_infer == test_rates)
                            test_obo.append(abs(test_infer - test_rates) <= 1)
                        else:
                            #train_cost.append(cost_batch)
                            nll_batch, auc_batch, _ = sess.run([cost_nll, auc, update_op], feed_dict={rate_batch: rates, infer: pred_batch})
                            proba_batch = sigmoid(pred_batch)
                            test_acc.append(np.round(proba_batch) == rates)
                            test_auc.append(auc_batch)
                            test_nll.append(nll_batch)
                    else:
                        l2_batch = sess.run(cost_l2, feed_dict={rate_batch: rates, infer: pred_batch})
                        test_se.append(np.power(rates - pred_batch, 2))

                end = time.time()
                test_rmse = np.sqrt(np.mean(test_se))
                test_macc = np.mean(test_acc)
                test_mobo = np.mean(test_obo)
                test_mauc = np.mean(test_auc)
                test_mnll = np.mean(test_nll)
                test_mcost = np.mean(test_cost)
                if DISCRETE:
                    if NB_CLASSES > 2:
                        print("{:3d} TRAIN(size={:d}/{:d}, macc={:f}, mobo={:f}, mcost={:f}) TEST(size={:d}, macc={:f}, mobo={:f}, mcost={:f}) {:f}(s)".format(
                            i // nb_batches,
                            len(train_users), len(train),
                            train_macc,
                            train_mobo,
                            train_mcost,
                            len(test),
                            test_macc,
                            test_mobo,
                            test_mcost,
                            end - start))
                    else:
                        print("{:3d} TRAIN(size={:d}/{:d}, macc={:f}, mauc={:f}, mnll={:f}) TEST(size={:d}, macc={:f}, mauc={:f}, mnll={:f}) {:f}(s)".format(
                            i // nb_batches,
                            len(train_users), len(train),
                            #train_rmse, # rmse={:f} 
                            train_macc, train_mauc, train_mnll,
                            len(test),
                            #test_rmse, # rmse={:f} 
                            test_macc, test_mauc, test_mnll,
                            end - start))
                else:
                    print("{:3d} TRAIN(size={:d}/{:d}, rmse={:f}) TEST(size={:d}, rmse={:f}) {:f}(s)".format(
                        i // nb_batches,
                        len(train_users), len(train),
                        train_rmse, # rmse={:f} 
                        #train_macc, train_mauc, train_mnll,
                        len(test),
                        test_rmse, # rmse={:f} 
                        #test_macc, test_mauc, test_mnll,
                        end - start))
                train_err_summary = make_scalar_summary("training_error", train_rmse)
                test_err_summary = make_scalar_summary("test_error", test_rmse)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end
        # print('thr', all_thresholds)

        # Save model
        print(os.path.join(BASE_DIR, 'fm.ckpt'))
        saver.save(sess, os.path.join(BASE_DIR, 'fm.ckpt'))


if __name__ == '__main__':
    df_train, df_val, df_test = dataio.get_data()
    print('Train', df_train.shape)
    print('Val', df_val.shape)
    print('Test', df_test.shape)
    svd(df_train, df_val)
    print("Done!")
