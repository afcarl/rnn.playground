import tflearn


def rnn_lstm_model(width, height):
    net = tflearn.input_data(shape=[None, height, width])
    net = tflearn.lstm(net, 128, return_seq=True)
    net = tflearn.lstm(net, 128)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def non_rnn_model(width, height):
    net = tflearn.input_data(shape=[None, height, width])
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def deeper_non_rnn_model(width, height):
    network = tflearn.input_data(shape=[None, height, width, 1])
    network = tflearn.conv_2d(network, 32, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2)
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2)
    network = tflearn.conv_2d(network, 128, 3, activation='relu')
    network = tflearn.conv_2d(network, 128, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2)
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 512, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 2, activation='softmax')
    network = tflearn.regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001, name="output1")
    return network


def non_rnn_fc_model(width, height):
    net = tflearn.input_data(shape=[None, height, width])
    net = tflearn.fully_connected(net, 128, activation='softmax')
    net = tflearn.fully_connected(net, 128, activation='softmax')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net
