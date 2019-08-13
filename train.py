import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
from utils.metrics.Nll import Nll
from utils.metrics.DocEmbSim import DocEmbSim
from utils.metrics.Bleu import Bleu
from utils.metrics.SelfBleu import SelfBleu
from utils.utils import *
from utils.ops import *
import texar as tx
from tensorflow.python import debug as tf_debug

EPS = 1e-10


# A function to initiate the graph and train the networks
def real_train(generator, discriminator, oracle_loader_train, oracle_loader_test, config):
    batch_size = config['batch_size']
    num_sentences = config['num_sentences']
    vocab_size = config['vocab_size']
    seq_len = config['seq_len']
    data_dir = config['data_dir']
    dataset = config['dataset']
    log_dir = config['log_dir']
    sample_dir = config['sample_dir']
    npre_epochs = config['npre_epochs']
    nadv_steps = config['nadv_steps']
    temper = config['temperature']
    adapt = config['adapt']
    save_dir = config['save_dir']
    ntest = config['ntest']
    dis_embed = config['dis_emb_dim']

    # filename
    pos_file = config['pos_file']
    neg_file = config['neg_file']
    test_pos_file = config['test_pos_file']
    test_neg_file = config['test_neg_file']
    voc_file = config['voc_file']

    gen_file = os.path.join(sample_dir, 'generator.txt')
    original_file = os.path.join(sample_dir, 'original.txt')
    gen_text_file = os.path.join(sample_dir, 'generator_text.txt')
    original_text_file = os.path.join(sample_dir, 'original_text.txt')
    csv_file = os.path.join(log_dir, 'experiment-log-rmcgan2.csv')

    # if dataset == 'yelp_15':
    #     test_file = os.path.join(data_dir, 'yelp_15/test_coco.txt')
    # elif dataset == 'yelp_40':
    #     test_file = os.path.join(data_dir, 'yelp_40/test_emnlp.txt')
    # else:
    #     raise NotImplementedError('Unknown dataset!')

    # create necessary directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # placeholder definitions
    # x_real = tf.placeholder(tf.int32, [batch_size, seq_len], name="x_real")  # tokens of oracle sequences
    batch_data = {}

    text_ids = tf.placeholder(tf.int32, shape=(batch_size, seq_len))
    text_keyword_id = tf.placeholder(tf.int32, shape=(batch_size, seq_len))
    text_keyword_length = tf.placeholder(tf.int32, shape=batch_size)
    labels = tf.placeholder(tf.int32, shape=batch_size)
    text_length = tf.placeholder(tf.int32, shape=batch_size)




    temperature = tf.Variable(1., trainable=False, name='temperature')


    x_real_onehot = tf.one_hot(text_ids, vocab_size)  # batch_size x seq_len x vocab_size
    #assert x_real_onehot.get_shape().as_list() == [batch_size, seq_len, vocab_size]

    # generator and discriminator outputs
    x_fake_apr, x_fake, g_pretrain_loss, gen_o = generator(text_ids, text_keyword_id, text_keyword_length,
                                                                   labels, text_length, temperature)

    x_fake_onehot = tf.one_hot(x_fake, vocab_size, 1.0, 0.0)
    loss_d_real, accu_d_real = discriminator(x_onehot=x_real_onehot, label=labels)
    loss_g_apr, _ = discriminator(x_onehot=x_fake_apr, label=(1-labels))
    _, accu_g_fake = discriminator(x_onehot=x_fake_onehot, label=(1 - labels))
    log_pg = tf.reduce_mean(tf.log(gen_o + EPS))

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    # Train ops
    g_loss = g_pretrain_loss+loss_g_apr
    g_pretrain_op, g_train_op, d_train_op = get_train_ops(config, g_pretrain_loss, g_loss, loss_d_real, global_step)

    # Record wall clock time
    time_diff = tf.placeholder(tf.float32)
    Wall_clock_time = tf.Variable(1., trainable=False)
    update_Wall_op = Wall_clock_time.assign_add(time_diff)

    # Temperature placeholder
    temp_var = tf.placeholder(tf.float32)
    update_temperature_op = temperature.assign(temp_var)

    # Loss summaries
    loss_summaries = [
        tf.summary.scalar('loss/discriminator', g_pretrain_loss),
        tf.summary.scalar('loss/g_loss_pre', g_loss),
        tf.summary.scalar('loss/g_loss', g_loss),
        tf.summary.scalar('loss/log_pg', log_pg),
        tf.summary.scalar('loss/Wall_clock_time', Wall_clock_time),
        tf.summary.scalar('loss/temperature', temperature),
    ]
    loss_summary_op = tf.summary.merge(loss_summaries)

    # Metric Summaries
    metrics_pl, metric_summary_op = get_metric_summary_op(config)


    def comput_BLEU(file_path):
        ori_sent = []
        gen_sent = []
        data = open(file_path, 'r').readlines()
        for i in range(len(data)):
            if (i+1)%2 == 0:
                ori_sent.append([data[i].strip('\n')])
            elif (i+1)%3 ==0:
                gen_sent.append(data[i].strip('\n'))

        hyps = np.array(gen_sent)
        refs = np.array(ori_sent)
        bleu, b1, b2, b3, b4 = tx.evals.corpus_bleu_moses(refs, hyps, return_all=True)
        return bleu, b1, b2, b3, b4

    def test(mode):
        # test

        global_acc = 0
        g_acc_real = 0

        for i in range(oracle_loader_test.num_batch):
            batch = oracle_loader_test.next_batch()
            x_gen, acc, acc_real = sess.run([x_fake, accu_g_fake, accu_d_real],
                                     feed_dict={text_ids: batch['text_ids'],
                                                text_keyword_id: batch['text_keyword_id'],
                                                text_keyword_length: batch[
                                                    'text_keyword_length'],
                                                labels: batch['labels'],
                                                text_length: batch['text_length']})

            text_samples(batch['text_ids'][:, 1:], x_gen, gen_file, original_file)
            # generate fake data and create batches
            gen_save_file = os.path.join(sample_dir, '{0}_samples_enpoch{1}_step{2}.txt'.format(mode, epoch, it))
            get_real_test_file(gen_file, gen_text_file, index_word_dict)
            get_real_test_file(original_file, original_text_file, index_word_dict)
            write_parallel_data(original_text_file, gen_text_file, batch['text_keyword'], gen_save_file)

            global_acc += acc
            g_acc_real += acc_real
        bleu, b1, b2, b3, b4 = comput_BLEU(gen_save_file)
        global_acc = global_acc / oracle_loader_test.num_batch
        g_acc_real = g_acc_real / oracle_loader_test.num_batch
        msg = 'test_fake_acc: %.4f' % global_acc + ', test_acc_real: %.4f' % g_acc_real + ', bleu: %.3f' % bleu \
              + ', b1: %.3f' % b1 + ', b2: %.3f' % b2 + ', b3: %.3f' % b3 + ', b4: %.3f' % b4
        log.write(msg)
        log.write('\n')
        print(msg)



    if config['is_train']:
        # ------------- initial the graph --------------
        with init_sess() as sess:

            # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)

            log = open(csv_file, 'w')
            sum_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'), sess.graph)

            # generate oracle data and create batches

            metrics = get_metrics(gen_file, original_file)

            oracle_loader_train.create_batches(neg_file, pos_file)
            print('train_batch_num: ' + str(oracle_loader_train.num_batch))
            oracle_loader_test.create_batches(test_neg_file, test_pos_file, shuff=False)
            index_word_dict = oracle_loader_train.index2word

            print('Start pre-training...')
            saver = tf.train.Saver()
            for epoch in range(npre_epochs):


                oracle_loader_train.reset_pointer()
                for it in range(oracle_loader_train.num_batch):
                    batch = oracle_loader_train.next_batch()
                    # pre-training
                    x_gen, g_pretrain_loss_np, d_loss_pre, accu_real, accu_fake, nll = pre_train_epoch(sess, g_pretrain_op, d_train_op, x_fake, accu_d_real, accu_g_fake,
                                                                g_pretrain_loss, loss_d_real, log_pg, text_ids, text_keyword_id,
                                                                text_keyword_length, labels, text_length, batch)


                    if (it % 20) == 0:
                        # msg = 'pre_gen_step:' + str(it) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
                        # print(msg)

                        # text_samples(batch['text_ids'][:, 1:], x_gen, gen_file, original_file)
                        #
                        # # generate fake data and create batches
                        # gen_save_file = os.path.join(sample_dir, 'pre_samples_enpoch{0}_step{1}.txt'.format(epoch, it))
                        # # a = generate_samples(transfer_sentence, gen_file)
                        # get_real_test_file(gen_file, gen_text_file, index_word_dict)
                        # get_real_test_file(original_file, original_text_file, index_word_dict)
                        # write_parallel_data(original_text_file, gen_text_file, batch['text_keyword'], gen_save_file)
                        #
                        # # write summaries
                        # scores = [metric.get_score() for metric in metrics]
                        # metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                        # sum_writer.add_summary(metrics_summary_str, epoch)

                        # msg = 'pre_gen_epoch:' + str(epoch) + ', step' + str(it) + ', g_pre_loss: %.4f' % g_pretrain_loss_np \
                        #       + ', d_loss_pre: %.4f' % d_loss_pre + ', acc_real: %.2f' % accu_real + ', accu_fake: % .2f' % accu_fake + ', NLL: %.4f' % nll
                        # metric_names = [metric.get_name() for metric in metrics]
                        # for (name, score) in zip(metric_names, scores):
                        #     msg += ', ' + name + ': %.4f' % score
                        msg = 'pre_gen_epoch:' + str(epoch) + ', step' + str(it) + ', g_pre_loss: %.4f' % g_pretrain_loss_np \
                              + ', d_loss_pre: %.4f' % d_loss_pre + ', acc_real: %.2f' % accu_real + ', accu_fake: %.2f' % accu_fake + ', NLL: %.4f' % nll
                        print(msg)
                        log.write(msg)
                        log.write('\n')

                    if (it % ntest)==0 and it != 0:
                        test('pre')
                        # if (it % 100) == 0:
                        #     # test
                        #     global_acc = 0
                        #     g_acc_real = 0
                        #
                        #     for it in range(oracle_loader_test.num_batch):
                        #         batch = oracle_loader_test.next_batch()
                        #         acc, acc_real = sess.run([accu_g_fake, accu_d_real],
                        #                                         feed_dict={text_ids: batch['text_ids'],
                        #                                                    text_keyword_id: batch['text_keyword_id'],
                        #                                                    text_keyword_length: batch[
                        #                                                        'text_keyword_length'],
                        #                                                    labels: batch['labels'],
                        #                                                    text_length: batch['text_length']})
                        #
                        #         global_acc += acc
                        #         g_acc_real += acc_real
                        #     global_acc = global_acc / oracle_loader_test.num_batch
                        #     g_acc_real = g_acc_real / oracle_loader_test.num_batch
                        #     print('test_fake_acc: %.4f' % global_acc)
                        #     print('test_acc_real: %.4f' % g_acc_real)

                saver.save(sess, os.path.join(save_dir, 'pre_train_model_epoch%d.ckpt' % epoch))
            # print('train_over')
            #
            # return

            print('Start adversarial training...')
            # progress = tqdm(range(nadv_steps))

            for it in range(oracle_loader_train.num_batch):
                oracle_loader_train.reset_pointer()
                # niter = sess.run(global_step)

                t0 = time.time()

                # adversarial training
                batch = oracle_loader_train.next_batch()
                sess.run(g_train_op, feed_dict={text_keyword_id: batch['text_keyword_id'],
                                                text_keyword_length: batch['text_keyword_length'],
                                                labels: batch['labels'],
                                                text_ids: batch['text_ids'],
                                                text_length: batch['text_length']})

                # for _ in range(config['gsteps']):
                #
                #     batch = oracle_loader.random_batch()
                #     sess.run(g_train_op, feed_dict={text_keyword_id: batch['text_keyword_id'],
                #                                     text_keyword_length: batch['text_keyword_length'],
                #                                     labels: batch['labels'],
                #                                     text_ids: batch['text_ids']})
                #     print()
                # for _ in range(config['dsteps']):
                #     batch = oracle_loader.random_batch()
                #     sess.run(d_train_op, feed_dict={text_keyword_id: batch['text_keyword_id'],
                #                                     text_keyword_length: batch['text_keyword_length'],
                #                                     labels: batch['labels'],
                #                                     text_ids: batch['text_ids']})

                t1 = time.time()
                sess.run(update_Wall_op, feed_dict={time_diff: t1 - t0})

                # temperature
                temp_var_np = get_fixed_temperature(temper, it, oracle_loader_train.num_batch, adapt)
                sess.run(update_temperature_op, feed_dict={temp_var: temp_var_np})

                batch = oracle_loader_train.random_batch()
                feed = {text_keyword_id: batch['text_keyword_id'],
                        text_keyword_length: batch['text_keyword_length'],
                        labels: batch['labels'],
                        text_ids: batch['text_ids'],
                        text_length: batch['text_length']}
                rec_loss, g_class_loss, acc_fake, loss_summary_str, nll = sess.run([g_pretrain_loss, loss_g_apr, accu_g_fake, loss_summary_op, log_pg], feed_dict=feed)
                sum_writer.add_summary(loss_summary_str, it)

                # sess.run(global_step_op)

                # progress.set_description('g_loss: %4.4f, d_loss: %4.4f' % (rec_loss, g_class_loss))

                # Test
                if (it % 20) == 0:

                    batch = oracle_loader_train.random_batch()
                    x_gen = sess.run(x_fake, feed_dict={text_ids: batch['text_ids'],
                                                       text_keyword_id: batch['text_keyword_id'],
                                                       text_keyword_length: batch['text_keyword_length'],
                                                       labels: batch['labels'],
                                                       text_length: batch['text_length']})
                    # # generate fake data and create batches
                    # gen_save_file = os.path.join(sample_dir, 'adv_samples_{:05d}.txt'.format(it))
                    # text_samples(batch['text_ids'][:, 1:], x_gen, gen_file, original_file)
                    #
                    # get_real_test_file(gen_file, gen_text_file, index_word_dict)
                    # get_real_test_file(original_file, original_text_file, index_word_dict)
                    # write_parallel_data(original_text_file, gen_text_file, batch['text_keyword'], gen_save_file)
                    # # get_real_test_file(gen_file, gen_save_file, index_word_dict)
                    #
                    # # write summaries
                    # scores = [metric.get_score() for metric in metrics]
                    # metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                    # sum_writer.add_summary(metrics_summary_str, it + config['npre_epochs'])

                    # msg = 'adv_step: ' + str(it) + ', rec_loss: %.4f' % rec_loss + ', g_class_loss: %.4f' % g_class_loss + ', acc_fake: %.2f' % acc_fake + 'NLL: %.4f' % nll
                    # # metric_names = [metric.get_name() for metric in metrics]
                    # for (name, score) in zip(metric_names, scores):
                    #     msg += ', ' + name + ': %.4f' % score
                    msg = 'adv_step: ' + str(it) + ', rec_loss: %.4f' % rec_loss + ', g_class_loss: %.4f' % g_class_loss \
                          + ', acc_fake: %.2f' % acc_fake + 'NLL: %.4f' % nll
                    print(msg)
                    log.write(msg)
                    log.write('\n')

                if (it % ntest) == 0 and it != 0:
                    test('adv')

            saver.save(sess, os.path.join(save_dir, 'adv_train_model_epoch%d.ckpt' % epoch))

    else:
        print("start_test")
        saver = tf.train.Saver()

        with init_sess() as sess:
            print('loading model')
            saver.restore(sess, config['checkpoint_file'])

            oracle_loader_test.create_batches(test_neg_file, test_pos_file, shuff=False)
            print('test_batch_num: ' + str(oracle_loader_test.num_batch))
            index_word_dict = oracle_loader_test.index2word

            global_acc = 0
            g_acc_real = 0
            with open('result/yelp_8.7/result2', 'w') as f:
                for it in range(oracle_loader_test.num_batch):
                    batch = oracle_loader_test.next_batch()
                    apr, x_gen, acc, acc_real = sess.run([x_fake_apr, x_fake, accu_g_fake, accu_d_real], feed_dict={text_ids: batch['text_ids'],
                                                                   text_keyword_id: batch['text_keyword_id'],
                                                                   text_keyword_length: batch['text_keyword_length'],
                                                                   labels: batch['labels'],
                                                                   text_length: batch['text_length']})

                    global_acc += acc
                    g_acc_real += acc_real
                    for i in range(batch_size):
                        m = ""
                        t = ""
                        k = ''
                        for z in batch["text"][i, 1:]:
                            if z == "<END>":
                                break
                            t = t + " " + z

                        for p in batch['text_keyword'][i]:
                            if p == "<END>":
                                break
                            k = k+" "+p
                        for j in x_gen[i, :]:
                            if j == 2:
                                break
                            m = m + " " + index_word_dict[j]
                        # f.write(k+'\n')
                        f.write(t.strip()+'\n')
                        f.write(m.strip()+'\n')
            # for it in range(oracle_loader.num_batch):
            #     batch = oracle_loader.next_batch()
            #     acc = sess.run([accu_d_real], feed_dict={text_ids: batch['text_ids'], labels: batch['labels']})
            #     global_acc += acc

            global_acc = global_acc / oracle_loader_test.num_batch
            g_acc_real = g_acc_real / oracle_loader_test.num_batch
            print('acc: %.4f' % global_acc)
            print('acc_real: %.4f' % g_acc_real)


# A function to get different GAN losses
def get_losses(loss_d_class, pre_train_loss, gen_o):
    # batch_size = config['batch_size']
    # gan_type = config['gan_type']
    #
    # if gan_type == 'standard':  # the non-satuating GAN loss
    #     d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=d_out_real, labels=tf.ones_like(d_out_real)
    #     ))
    #     d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
    #     ))
    #     d_loss = d_loss_real + d_loss_fake
    #
    #     g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=d_out_fake, labels=tf.ones_like(d_out_fake)
    #     ))
    #
    # elif gan_type == 'JS':  # the vanilla GAN loss
    #     d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=d_out_real, labels=tf.ones_like(d_out_real)
    #     ))
    #     d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
    #     ))
    #     d_loss = d_loss_real + d_loss_fake
    #
    #     g_loss = -d_loss_fake
    #
    # elif gan_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
    #     d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=d_out_real, labels=tf.ones_like(d_out_real)
    #     ))
    #     d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
    #     ))
    #     d_loss = d_loss_real + d_loss_fake
    #
    #     g_loss = tf.reduce_mean(-d_out_fake)
    #
    # elif gan_type == 'hinge':  # the hinge loss
    #     d_loss_real = tf.reduce_mean(tf.nn.relu(sentiment.1.0 - d_out_real))
    #     d_loss_fake = tf.reduce_mean(tf.nn.relu(sentiment.1.0 + d_out_fake))
    #     d_loss = d_loss_real + d_loss_fake
    #
    #     g_loss = -tf.reduce_mean(d_out_fake)
    #
    # elif gan_type == 'tv':  # the total variation distance
    #     d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
    #     g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))
    #
    # elif gan_type == 'wgan-gp':  # WGAN-GP
    #     d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
    #     GP = gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config)
    #     d_loss += GP
    #
    #     g_loss = -tf.reduce_mean(d_out_fake)
    #
    # elif gan_type == 'LS':  # LS-GAN
    #     d_loss_real = tf.reduce_mean(tf.squared_difference(d_out_real, sentiment.1.0))
    #     d_loss_fake = tf.reduce_mean(tf.square(d_out_fake))
    #     d_loss = d_loss_real + d_loss_fake
    #
    #     g_loss = tf.reduce_mean(tf.squared_difference(d_out_fake, sentiment.1.0))
    #
    # elif gan_type == 'RSGAN':  # relativistic standard GAN
    #     d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=d_out_real - d_out_fake, labels=tf.ones_like(d_out_real)
    #     ))
    #     g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=d_out_fake - d_out_real, labels=tf.ones_like(d_out_fake)
    #     ))
    #
    # else:
    #     raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)

    log_pg = tf.reduce_mean(tf.log(gen_o + EPS))  # [sentiment.1], measures the log p_g(x)

    return log_pg, pre_train_loss, loss_d_class


# A function to calculate the gradients and get training operations
def get_train_ops(config, g_pretrain_loss, g_loss, d_loss_real, global_step):
    optimizer_name = config['optimizer']
    nadv_steps = config['nadv_steps']
    d_lr = config['d_lr']
    gpre_lr = config['gpre_lr']
    gadv_lr = config['gadv_lr']

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    grad_clip = 5.0  # keep the same with the previous setting

    # generator pre-training
    pretrain_opt = tf.train.AdamOptimizer(gpre_lr, beta1=0.9, beta2=0.999)
    pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(g_pretrain_loss, g_vars), grad_clip)  # gradient clipping
    g_pretrain_op = pretrain_opt.apply_gradients(zip(pretrain_grad, g_vars))

    # decide if using the weight decaying
    if config['decay']:
        d_lr = tf.train.exponential_decay(d_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)
        gadv_lr = tf.train.exponential_decay(gadv_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)

    # Adam optimizer
    if optimizer_name == 'adam':
        d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=0.9, beta2=0.999)
        g_optimizer = tf.train.AdamOptimizer(gadv_lr, beta1=0.9, beta2=0.999)
        temp_optimizer = tf.train.AdamOptimizer(1e-2, beta1=0.9, beta2=0.999)

    # RMSProp optimizer
    elif optimizer_name == 'rmsprop':
        d_optimizer = tf.train.RMSPropOptimizer(d_lr)
        g_optimizer = tf.train.RMSPropOptimizer(gadv_lr)
        temp_optimizer = tf.train.RMSPropOptimizer(1e-2)

    else:
        raise NotImplementedError

    # gradient clipping
    g_grads, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars), grad_clip)
    g_train_op = g_optimizer.apply_gradients(zip(g_grads, g_vars))

    print('len of g_grads without None: {}'.format(len([i for i in g_grads if i is not None])))
    print('len of g_grads: {}'.format(len(g_grads)))

    # gradient clipping
    d_grads, _ = tf.clip_by_global_norm(tf.gradients(d_loss_real, d_vars), grad_clip)
    d_train_op = d_optimizer.apply_gradients(zip(d_grads, d_vars))

    return g_pretrain_op, g_train_op, d_train_op


# A function to get various evaluation metrics
def get_metrics(gen_file, original_file):
    # set up evaluation metric
    metrics = []
    for i in range(1, 5):
        selfbleu = SelfBleu(original_text=original_file, transfer_text=gen_file, gram=i, name='selfbleu' + str(i))
        metrics.append(selfbleu)

    # if config['nll_gen']:
    #     nll_gen = Nll(oracle_loader, g_pretrain_loss, text_ids, sess, name='nll_gen')
    #     metrics.append(nll_gen)
    # if config['doc_embsim']:
    #     doc_embsim = DocEmbSim(test_file, gen_file, config['vocab_size'], name='doc_embsim')
    #     metrics.append(doc_embsim)
    # if config['bleu']:
    #     for i in range(2, 6):
    #         bleu = Bleu(test_text=gen_file, real_text=test_file, gram=i, name='bleu' + str(i))
    #         metrics.append(bleu)
    # if config['selfbleu']:
    #     for i in range(2, 6):
    #         selfbleu = SelfBleu(test_text=gen_file, gram=i, name='selfbleu' + str(i))
    #         metrics.append(selfbleu)
    return metrics


# A function to get the summary for each metric
def get_metric_summary_op(config):
    metrics_pl = []
    metrics_sum = []

    if config['nll_gen']:
        nll_gen = tf.placeholder(tf.float32)
        metrics_pl.append(nll_gen)
        metrics_sum.append(tf.summary.scalar('metrics/nll_gen', nll_gen))

    if config['doc_embsim']:
        doc_embsim = tf.placeholder(tf.float32)
        metrics_pl.append(doc_embsim)
        metrics_sum.append(tf.summary.scalar('metrics/doc_embsim', doc_embsim))

    if config['bleu']:
        for i in range(2, 6):
            temp_pl = tf.placeholder(tf.float32, name='bleu{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/bleu{}'.format(i), temp_pl))

    if config['selfbleu']:
        for i in range(1, 5):
            temp_pl = tf.placeholder(tf.float32, name='selfbleu{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/selfbleu{}'.format(i), temp_pl))

    metric_summary_op = tf.summary.merge(metrics_sum)
    return metrics_pl, metric_summary_op


# A function to set up different temperature control policies
def get_fixed_temperature(temper, i, N, adapt):
    if adapt == 'no':
        temper_var_np = temper  # no increase
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1)**2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        raise Exception("Unknown adapt type!")

    return 1/temper_var_np
