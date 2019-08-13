from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from utils.ops import *
import texar as tx
from Gan_architecture import lstm_config
from texar.modules import UnidirectionalRNNEncoder, MLPTransformConnector, AttentionRNNDecoder, \
    Conv1DClassifier,GumbelSoftmaxEmbeddingHelper
from texar.utils import transformer_utils
import numpy as np

hparams = tx.HParams(lstm_config.model, None)
#The generator network based on the Relational Memory
def generator(text_ids, text_keyword_id, text_keyword_length, labels, text_length, temperature, vocab_size, batch_size,
              seq_len, gen_emb_dim, mem_slots, head_size, num_heads,hidden_dim, start_token):



    # Source word embedding
    src_word_embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=hparams.embedder)
    src_word_embeds = src_word_embedder(text_keyword_id)


    encoder = UnidirectionalRNNEncoder(hparams=hparams.encoder)
    enc_outputs, final_state = encoder(inputs=src_word_embeds,
                             sequence_length=text_keyword_length)

    # modify sentiment label
    label_connector = MLPTransformConnector(output_size=hparams.dim_c)
    state_connector = MLPTransformConnector(output_size=700)

    labels = tf.to_float(tf.reshape(labels, [batch_size, 1]))
    c = label_connector(labels)
    c_ = label_connector(1-labels)
    h = tf.concat([c, final_state], axis=1)
    h_ = tf.concat([c_, final_state], axis=1)

    state = state_connector(h)
    state_ = state_connector(h_)


    decoder = AttentionRNNDecoder(memory=enc_outputs,
                                  memory_sequence_length=text_keyword_length,
                                  cell_input_fn=lambda inputs, attention: inputs,
                                  vocab_size=vocab_size,
                                  hparams=hparams.decoder)

    # For training
    g_outputs, _, _ = decoder(
        initial_state=state, inputs=text_ids,
        embedding=src_word_embedder, sequence_length=tf.convert_to_tensor(np.array([(seq_len-1) for i in range(batch_size)], dtype=np.int32)))
    # e = g_outputs.cell_output


    start_tokens = np.ones(batch_size, int)
    end_token = int(2)
    # Greedy decoding, used in eval
    outputs_, _, length_ = decoder(
        decoding_strategy='infer_greedy', initial_state=state_,
        embedding=src_word_embedder, start_tokens=start_tokens, end_token=end_token)

    pretrain_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=text_ids[:, 1:],
        logits=g_outputs.logits,
        sequence_length=text_length-1,
        average_across_timesteps=True,
        sum_over_timesteps=False)

    # # Gumbel-softmax decoding, used in training
    gumbel_helper = GumbelSoftmaxEmbeddingHelper(
        src_word_embedder.embedding, start_tokens, end_token, temperature)

    gumbel_outputs, _, sequence_lengths = decoder(
        helper=gumbel_helper, initial_state=state_)

    # max_index = tf.argmax(gumbel_outputs.logits, axis=2)


    gen_o = tf.reduce_sum(tf.reduce_max(outputs_.logits, axis=2), 1)

    return gumbel_outputs.logits, outputs_.sample_id, pretrain_loss, gen_o
    # return outputs_.sample_id, pretrain_loss, gen_o


# The discriminator network based on the CNN classifier
def discriminator(x_onehot, batch_size, label, vocab_size, seq_len, dis_emb_dim, num_rep, sn):


    # while load_wordvec:
    #     embed = read_wordvec('data/glove.twitter.27B.100d.txt', index2word)
    #     load_wordvec = False

    # d_embeddings = tf.get_variable('d_emb', shape=[vocab_size, dis_emb_dim],
    #                                initializer=tf.constant_initializer(embed))
    vocab_tensor = [i for i in range(vocab_size)]
    clas_embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=hparams.embedder)
    embedding = clas_embedder(vocab_tensor)

    # clas_embedder = WordEmbedder(vocab_size=vocab.size,
    #                              hparams=self._hparams.embedder)

    classifier = Conv1DClassifier(hparams=hparams.classifier)

    input_x_re = tf.reshape(x_onehot, [-1, vocab_size])
    # emb_x_re = clas_embedder(input_x_re)
    emb_x_re = tf.matmul(input_x_re, embedding)
    x = tf.reshape(emb_x_re, [batch_size, -1, 100])

    clas_logits, clas_preds = classifier(
        inputs=x,
        sequence_length=emb_x_re.get_shape()[1].value)

    loss_d_clas = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(label), logits=clas_logits)
    loss_d_clas = tf.reduce_mean(loss_d_clas)
    accu_d = tx.evals.accuracy(labels=label, preds=clas_preds)

    return loss_d_clas, accu_d

# # The discriminator network based on the CNN classifier
# def discriminator(x_onehot, batch_size, embed, seq_len, vocab_size, dis_emb_dim, num_rep, sn, load_wordvec=False):
#     # get the embedding dimension for each presentation
#     emb_dim_single = int(dis_emb_dim / num_rep)
#     assert isinstance(emb_dim_single, int) and emb_dim_single > 0
#
#     filter_sizes = [2, 3, 4, 5]
#     num_filters = [300, 300, 300, 300]
#     dropout_keep_prob = 0.75
#
#     # if load_wordvec:
#     #     embed = read_wordvec('data/glove.twitter.27B.100d.txt', id2word)
#
#
#     d_embeddings = tf.get_variable('d_emb', shape=[vocab_size, dis_emb_dim],
#                                    initializer=tf.constant_initializer(embed))
#     input_x_re = tf.reshape(x_onehot, [-sentiment.1, vocab_size])
#     emb_x_re = tf.matmul(input_x_re, d_embeddings)
#     emb_x = tf.reshape(emb_x_re, [batch_size, seq_len, dis_emb_dim])  # batch_size x seq_len x dis_emb_dim
#
#     emb_x_expanded = tf.expand_dims(emb_x, -sentiment.1)  # batch_size x seq_len x dis_emb_dim x sentiment.1
#     print('shape of emb_x_expanded: {}'.format(emb_x_expanded.get_shape().as_list()))
#
#     # Create a convolution + maxpool layer for each filter size
#     pooled_outputs = []
#     for filter_size, num_filter in zip(filter_sizes, num_filters):
#         conv = conv2d(emb_x_expanded, num_filter, k_h=filter_size, k_w=emb_dim_single,
#                       d_h=sentiment.1, d_w=emb_dim_single, sn=sn, stddev=None, padding='VALID',
#                       scope="conv-%s" % filter_size)  # batch_size x (seq_len-k_h+sentiment.1) x num_rep x num_filter
#         out = tf.nn.relu(conv, name="relu")
#         pooled = tf.nn.max_pool(out, ksize=[sentiment.1, seq_len - filter_size + sentiment.1, sentiment.1, sentiment.1],
#                                 strides=[sentiment.1, sentiment.1, sentiment.1, sentiment.1], padding='VALID',
#                                 name="pool")  # batch_size x sentiment.1 x num_rep x num_filter
#         pooled_outputs.append(pooled)
#
#     # Combine all the pooled features
#     num_filters_total = sum(num_filters)
#     h_pool = tf.concat(pooled_outputs, 3)  # batch_size x sentiment.1 x num_rep x num_filters_total
#     print('shape of h_pool: {}'.format(h_pool.get_shape().as_list()))
#     h_pool_flat = tf.reshape(h_pool, [-sentiment.1, num_filters_total])
#
#     # Add highway
#     h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[sentiment.1], sentiment.1, 0)  # (batch_size*num_rep) x num_filters_total
#
#     # Add dropout
#     h_drop = tf.nn.dropout(h_highway, dropout_keep_prob, name='dropout')
#
#     # fc
#     fc_out = linear(h_drop, output_size=100, use_bias=True, sn=sn, scope='fc')
#     logits = linear(fc_out, output_size=sentiment.1, use_bias=True, sn=sn, scope='logits')
#     logits = tf.squeeze(logits, -sentiment.1)  # batch_size*num_rep
#
#     return logits