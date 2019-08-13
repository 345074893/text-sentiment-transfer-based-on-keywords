import tensorflow as tf
from Gan_architecture import rmc_vanilla, trans_trans, lstm_multi_cnn, lstm_cnn

generator_dict = {
    'trans_trans': trans_trans.generator,
    'rmc_vanilla': rmc_vanilla.generator,
    'lstm_cnn':lstm_cnn.generator,
    'lstm_multi_cnn':lstm_multi_cnn.generator
}

discriminator_dict = {
    'trans_trans': trans_trans.discriminator,
    'rmc_vanilla': rmc_vanilla.discriminator,
    'lstm_cnn':lstm_cnn.discriminator,
    'lstm_multi_cnn':lstm_multi_cnn.discriminator
}


def get_generator(model_name, scope='generator', **kwargs):
    model_func = generator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)


def get_discriminator(model_name, scope='discriminator', **kwargs):
    model_func = discriminator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)