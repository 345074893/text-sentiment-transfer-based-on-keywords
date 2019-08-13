from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from utils.ops import *
import texar as tx
from Gan_architecture import trans_config
from texar.modules import TransformerEncoder, TransformerDecoder, MLPTransformConnector, GumbelSoftmaxEmbeddingHelper
from texar.utils import transformer_utils
import numpy as np


#The generator network based on the Relational Memory
def generator(text_ids, text_keyword_id, text_keyword_length, labels, text_length, temperature, vocab_size, batch_size,
              seq_len, gen_emb_dim, mem_slots, head_size, num_heads,hidden_dim, start_token):


    is_target = tf.to_float(tf.not_equal(text_ids[:, 1:], 0))

    # Source word embedding
    src_word_embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=trans_config.emb)
    src_word_embeds = src_word_embedder(text_keyword_id)
    src_word_embeds = src_word_embeds * trans_config.hidden_dim ** 0.5

    # Position embedding (shared b/w source and target)
    pos_embedder = tx.modules.SinusoidsPositionEmbedder(
        position_size=seq_len,
        hparams=trans_config.position_embedder_hparams)
    # src_seq_len = batch_data['text_keyword_length']
    src_pos_embeds = pos_embedder(sequence_length=seq_len)

    src_input_embedding = src_word_embeds + src_pos_embeds

    encoder = TransformerEncoder(hparams=trans_config.encoder)
    encoder_output = encoder(inputs=src_input_embedding,
                             sequence_length=text_keyword_length)

    # modify sentiment label
    label_connector = MLPTransformConnector(output_size=trans_config.hidden_dim)

    labels = tf.to_float(tf.reshape(labels, [-1, 1]))
    c = tf.reshape(label_connector(labels), [batch_size, 1, 512])
    c_ = tf.reshape(label_connector(1-labels), [batch_size, 1, 512])
    encoder_output = tf.concat([c, encoder_output[:, 1:, :]], axis=1)
    encoder_output_ = tf.concat([c_, encoder_output[:, 1:, :]], axis=1)


    # The decoder ties the input word embedding with the output logit layer.
    # As the decoder masks out <PAD>'s embedding, which in effect means
    # <PAD> has all-zero embedding, so here we explicitly set <PAD>'s embedding
    # to all-zero.
    tgt_embedding = tf.concat(
        [tf.zeros(shape=[1, src_word_embedder.dim]),
         src_word_embedder.embedding[1:, :]],
        axis=0)
    tgt_embedder = tx.modules.WordEmbedder(tgt_embedding)
    tgt_word_embeds = tgt_embedder(text_ids)
    tgt_word_embeds = tgt_word_embeds * trans_config.hidden_dim ** 0.5

    tgt_seq_len = text_length
    tgt_pos_embeds = pos_embedder(sequence_length=tgt_seq_len)

    tgt_input_embedding = tgt_word_embeds + tgt_pos_embeds

    _output_w = tf.transpose(tgt_embedder.embedding, (1, 0))

    decoder = TransformerDecoder(vocab_size=vocab_size,
                                 output_layer=_output_w,
                                 hparams=trans_config.decoder)
    # For training
    outputs = decoder(
        memory=encoder_output,
        memory_sequence_length=text_keyword_length,
        inputs=tgt_input_embedding,
        decoding_strategy='train_greedy',
        mode=tf.estimator.ModeKeys.TRAIN
    )

    mle_loss = transformer_utils.smoothing_cross_entropy(
        outputs.logits[:, :-1, :], text_ids[:, 1:], vocab_size, trans_config.loss_label_confidence)
    pretrain_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)



    # Gumbel-softmax decoding, used in training
    start_tokens = np.ones(batch_size, int)
    end_token = int(2)
    gumbel_helper = GumbelSoftmaxEmbeddingHelper(
        tgt_embedding, start_tokens, end_token, temperature)

    gumbel_outputs, sequence_lengths = decoder(
        memory=encoder_output_,
        memory_sequence_length=text_keyword_length,
        helper=gumbel_helper
        )

    # max_index = tf.argmax(gumbel_outputs.logits, axis=2)
    # gen_x_onehot_adv = tf.one_hot(max_index, vocab_size, sentiment.1.0, 0.0)

    gen_o = tf.reduce_sum(tf.reduce_max(gumbel_outputs.logits, axis=2))

    return gumbel_outputs.logits, gumbel_outputs.sample_id, pretrain_loss, gen_o


# The discriminator network based on the CNN classifier
def discriminator(x_onehot, load_wordvec, id2word, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn):

    # get the embedding dimension for each presentation
    emb_dim_single = int(dis_emb_dim / num_rep)
    assert isinstance(emb_dim_single, int) and emb_dim_single > 0

    filter_sizes = [2, 3, 4, 5]
    num_filters = [300, 300, 300, 300]
    dropout_keep_prob = 0.75

    while load_wordvec:
        embed = read_wordvec('data/glove.twitter.27B.100d.txt', id2word)
        load_wordvec = False

    d_embeddings = tf.get_variable('d_emb', shape=[vocab_size, dis_emb_dim],
                                    initializer=tf.constant_initializer(embed))

    input_x_re = tf.reshape(x_onehot, [-1, vocab_size])
    emb_x_re = tf.matmul(input_x_re, d_embeddings)
    emb_x = tf.reshape(emb_x_re, [batch_size, seq_len, dis_emb_dim])  # batch_size x seq_len x dis_emb_dim

    emb_x_expanded = tf.expand_dims(emb_x, -1)  # batch_size x seq_len x dis_emb_dim x sentiment.1
    print('shape of emb_x_expanded: {}'.format(emb_x_expanded.get_shape().as_list()))

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for filter_size, num_filter in zip(filter_sizes, num_filters):
        conv = conv2d(emb_x_expanded, num_filter, k_h=filter_size, k_w=emb_dim_single,
                      d_h=1, d_w=emb_dim_single, sn=sn, stddev=None, padding='VALID',
                      scope="conv-%s" % filter_size)  # batch_size x (seq_len-k_h+sentiment.1) x num_rep x num_filter
        out = tf.nn.relu(conv, name="relu")
        pooled = tf.nn.max_pool(out, ksize=[1, seq_len - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1], padding='VALID',
                                name="pool")  # batch_size x sentiment.1 x num_rep x num_filter
        pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = sum(num_filters)
    h_pool = tf.concat(pooled_outputs, 3)  # batch_size x sentiment.1 x num_rep x num_filters_total
    print('shape of h_pool: {}'.format(h_pool.get_shape().as_list()))
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add highway
    h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)  # (batch_size*num_rep) x num_filters_total

    # Add dropout
    h_drop = tf.nn.dropout(h_highway, dropout_keep_prob, name='dropout')

    # fc
    fc_out = linear(h_drop, output_size=100, use_bias=True, sn=sn, scope='fc')
    is_real_logits = linear(fc_out, output_size=1, use_bias=True, sn=sn, scope='logits')
    # sentiment_logits = linear(fc_out, output_size=2, use_bias=True, sn=sn, scope='sentiment_logits')
    #
    # sentiment_prob = tf.nn.softmax(sentiment_logits, axis=2)
    # sentimen_class = tf.arg_max(sentiment_prob)

    is_real_logits = tf.squeeze(is_real_logits, -1)  # batch_size*num_rep

    # return is_real_logits, sentiment_prob, sentimen_class
    return is_real_logits