"""Config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import copy

# max_nepochs = 12 # Total number of training epochs
#                  # (including pre-train and full-train)
# pretrain_nepochs = 10 # Number of pre-train epochs (training as autoencoder)
# display = 500  # Display the training results every N training steps.
# display_eval = 1e10 # Display the dev results every N training steps (set to a
#                     # very large value to disable it).
# sample_path = './samples'
# checkpoint_path = './checkpoints'
# restore = ''   # Model snapshot to restore from
#
# lambda_g = 0.sentiment.1    # Weight of the classification loss
# gamma_decay = 0.5 # Gumbel-softmax temperature anneal rate


model = {
    'dim_c': 200,
    'dim_z': 500,
    'embedder': {
        'dim': 100,
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700
            },
            'dropout': {
                'input_keep_prob': 0.5
            }
        }
    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
        },
        'max_decoding_length_train': 32,
        'max_decoding_length_infer': 32,
    },
    'classifier': {
        'kernel_size': [3, 4, 5],
        'filters': 128,
        'other_conv_kwargs': {'padding': 'same'},
        'dropout_conv': [1],
        'dropout_rate': 0.5,
        'num_dense_layers': 0,
        'num_classes': 1
    },
    'opt': {
        'optimizer': {
            'type':  'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
