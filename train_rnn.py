from __future__ import division, print_function, absolute_import

import PIL.Image as Image
import numpy as np
import tflearn
from models import deeper_non_rnn_model, non_rnn_model, rnn_lstm_model, non_rnn_fc_model
from paths_ds import PathDataSet

WIDTH = 20
HEIGHT = 40
NUM_SAMPLES = 20000
NUM_EPOCHS = 3

myDs = PathDataSet(num_samples=NUM_SAMPLES)
X, Y = myDs.generate_ds()
X = np.array([np.array(Image.fromarray(img).resize((WIDTH,HEIGHT))) for img in X])
X = np.reshape(X, (-1, HEIGHT, WIDTH))


# try rnn model
model = rnn_lstm_model(WIDTH, HEIGHT)
tensor_board_path = "/tmp/rnn_playground/rnn"

# try non rnn model
'''
model = non_rnn_model(WIDTH, HEIGHT)
tensor_board_path = "/tmp/rnn_playground/non_rnn"
'''

# try deep conv arch
'''
X = np.reshape(X, (-1, HEIGHT, WIDTH, 1))
model = deeper_non_rnn_model(WIDTH, HEIGHT)
tensor_board_path = "/tmp/rnn_playground/deep"

'''

# try fc model
'''
model = non_rnn_fc_model(WIDTH, HEIGHT)
tensor_board_path = "/tmp/rnn_playground/fc
'''

model = tflearn.DNN(model, tensorboard_verbose=2, tensorboard_dir=tensor_board_path)
model.fit(X, Y, n_epoch=NUM_EPOCHS, validation_set=0.1, show_metric=True, shuffle=True)
