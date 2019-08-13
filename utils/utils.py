import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pprint
from utils.text_process import *

pp = pprint.PrettyPrinter()


# def generate_samples(sess, gen_x, batch_size, generated_num, output_file=None, get_code=True):
#     # Generate Samples
#     generated_samples = []
#     for _ in range(int(generated_num / batch_size)):
#         generated_samples.extend(sess.run(gen_x))
#     codes = list()
#     if output_file is not None:
#         with open(output_file, 'w') as fout:
#             for sent in generated_samples:
#                 buffer = ' '.join([str(x) for x in sent]) + '\n'
#                 fout.write(buffer)
#                 if get_code:
#                     codes.append(sent)
#         return np.array(codes)
#     codes = ""
#     for sent in generated_samples:
#         buffer = ' '.join([str(x) for x in sent]) + '\n'
#         codes += buffer
#     return codes

def text_samples(original_x, gen_x, gen_file, original_file, get_code=True):
    # Generate Samples
    codes = list()
    with open(original_file, 'w') as fout:
        for i in range(len(original_x)):
            buffer = ' '.join([str(x) for x in original_x[i]]) + '\n'
            fout.write(buffer)
            # if get_code:
            #     codes.append(sent)
    with open(gen_file, 'w') as f:
        for i in range(len(gen_x)):
            buffer = ' '.join([str(x) for x in gen_x[i]]) + '\n'
            f.write(buffer)

    # return np.array(codes)
    # codes = ""
    # for sent in gen_x:
    #     buffer = ' '.join([str(x) for x in sent]) + '\n'
    #     codes += buffer
    # return codes



def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


def pre_train_epoch(sess, g_pretrain_op, d_train_op, x_fake, accu_d_real, accu_d_fake, g_pretrain_loss, loss_d_real, log_pg, text_ids, text_keyword_id, text_keyword_length, labels,
                    text_length, batch):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    # oracle_loader.reset_pointer()

    _, _, x_gen, g_loss, loss_d, accu_real, accu_fake, nll = sess.run([g_pretrain_op, d_train_op, x_fake, g_pretrain_loss, loss_d_real, accu_d_real, accu_d_fake, log_pg], feed_dict={text_ids: batch['text_ids'],
                                                                                     text_keyword_id: batch[
                                                                                         'text_keyword_id'],
                                                                                     text_keyword_length: batch[
                                                                                         'text_keyword_length'],
                                                                                     labels: batch['labels'],
                                                                                     text_length: batch['text_length']})
    return x_gen, g_loss, loss_d, accu_real, accu_fake, nll

    # for it in range(oracle_loader.num_batch):
    #     batch = oracle_loader.next_batch()
    #     _, x_gen, g_loss = sess.run([g_pretrain_op, x_fake, g_pretrain_loss], feed_dict={text_ids: batch['text_ids'],
    #                                                                       text_keyword_id: batch['text_keyword_id'],
    #                                                                       text_keyword_length: batch['text_keyword_length'],
    #                                                                       labels: batch['labels'],
    #                                                                       text_length: batch['text_length']})
        # if (it % 100) == 0:
        #     msg = 'pre_gen_step:' + str(it) + ', g_pre_loss: %.4f' % g_loss
        #     print(msg)
        #
        # if it == oracle_loader.num_batch-sentiment.1:
        #     text_samples(batch['text_ids'][:, sentiment.1:], x_gen, gen_file, original_file)

    #     supervised_g_losses.append(g_loss)
    #
    # return np.mean(supervised_g_losses)


def plot_csv(csv_file, pre_epoch_num, metrics, method):
    names = [str(i) for i in range(len(metrics) + 1)]
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=0, skip_footer=0, names=names)
    for idx in range(len(metrics)):
        metric_name = metrics[idx].get_name()
        plt.figure()
        plt.plot(data[names[0]], data[names[idx + 1]], color='r', label=method)
        plt.axvline(x=pre_epoch_num, color='k', linestyle='--')
        plt.xlabel('training epochs')
        plt.ylabel(metric_name)
        plt.legend()
        plot_file = os.path.join(os.path.dirname(csv_file), '{}_{}.pdf'.format(method, metric_name))
        print(plot_file)
        plt.savefig(plot_file)


def get_oracle_file(data_file, oracle_file, seq_len):
    tokens = get_tokenlized(data_file)
    word_set = get_word_list(tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)
    with open(oracle_file, 'w') as outfile:
        outfile.write(text_to_code(tokens, word_index_dict, seq_len))

    return index_word_dict


def get_real_test_file(generator_file, gen_save_file, iw_dict):
    codes = get_tokenlized(generator_file)
    with open(gen_save_file, 'w') as outfile:
        outfile.write(code_to_text(codes=codes, dictionary=iw_dict))

def write_parallel_data(original_file, generater_file, text_keyword_pos, save_file):
    original_sentences = open(original_file, 'r').readlines()
    generater_sentences = open(generater_file, 'r').readlines()
    with open(save_file, 'a') as f:
        for i in range(len(original_sentences)):
            for j in text_keyword_pos[i]:
                if j == '<END>':
                    break
                f.write(j+ ' ')
                # f.write('(')
                # for word in j:
                #     f.write(word+', ')
                # f.write('), ')
            f.write('\n')

            f.write(original_sentences[i])
            f.write(generater_sentences[i])



