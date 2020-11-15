import tensorflow as tf
import numpy as np
import loadData


def one_hot(label, num_class):
    label = np.array(label)
    num_label = label.shape[0]
    index = np.arange(num_label) * num_class
    out = np.zeros((num_label, num_class))
    out.flat[index + label.ravel()] = 1
    return out


def model_1(x_in, x_out, hidden_size, seq_len_in, seq_len_out):
    with tf.variable_scope('LSTM_1', reuse=tf.AUTO_REUSE):
        cell_in_1 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
        outputs_in, states_in = tf.nn.dynamic_rnn(
            cell=cell_in_1,
            inputs=x_in,
            sequence_length=seq_len_in,
            time_major=False,
            dtype=tf.float32,
        )
    with tf.variable_scope('LSTM_2', reuse=tf.AUTO_REUSE):
        cell_out_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
        outputs_out, states_out = tf.nn.dynamic_rnn(
            cell=cell_out_1,
            inputs=x_out,
            sequence_length=seq_len_out,
            time_major=False,
            dtype=tf.float32,
        )
    return outputs_in, outputs_out


def model_2(outputs_in, outputs_out, hidden_size):
    # params:
    # output:(batch_size,seq_len,hidden_lstm)
    # return:
    # result:(batch_size,)
    with tf.name_scope('coAttention'):
        dense = tf.layers.Dense(hidden_size, activation='relu')
        x = dense(outputs_in)
        # matrix:(batch_size,seq_in_len,seq_out_len)
        # missing decode
        matrix = tf.matmul(x, tf.transpose(outputs_out, perm=[0, 2, 1]))
        input_weight = tf.nn.softmax(matrix, dim=1)
        output_weight = tf.nn.softmax(matrix, dim=-1)
        # (batch_size,seq_in_len,hidden_lstm)
        coat_in = tf.matmul(input_weight, outputs_out)
        # (batch_size,seq_out_len,hidden_lstm)
        coat_out = tf.matmul(tf.transpose(output_weight, perm=[0, 2, 1]),
                             outputs_in)
        co_in_result = tf.concat((coat_in, outputs_in), -1)
        co_out_result = tf.concat((coat_out, outputs_out), -1)
    return co_in_result, co_out_result


def model_3(co_in_result, co_out_result, hidden_size):
    with tf.variable_scope('LSTM_3', reuse=tf.AUTO_REUSE):
        size = co_in_result.shape[-1]
        cell_in_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=size)
        output_final_in, state_final_in = tf.nn.dynamic_rnn(
            cell=cell_in_2,
            inputs=co_in_result,
            time_major=False,
            dtype=tf.float32)
    with tf.variable_scope('LSTM_4', reuse=tf.AUTO_REUSE):
        size = co_out_result.shape[-1]
        cell_out_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=size)
        output_final_out, state_final_out = tf.nn.dynamic_rnn(
            cell=cell_out_2,
            inputs=co_out_result,
            time_major=False,
            dtype=tf.float32)
    return output_final_in, output_final_out


def main():
    x_train_in, x_train_out, x_test_in, x_test_out, y_train, y_test, seq_len_in, seq_len_out = loadData.load_array(
        train_file='sort_data/TrainSamples.txt',
        test_file='sort_data/TestSamples.txt',
        train_label='sort_data/TrainLabel.txt',
        test_label='sort_data/TestLabel.txt')
    seq_train_in = seq_len_in[0:len(x_train_in)]
    seq_test_in = seq_len_in[len(x_train_in):]
    seq_train_out = seq_len_out[0:len(x_train_out)]
    seq_test_out = seq_len_out[len(x_train_out):]
    batch_size = 64
    embed_len_in = x_train_in.shape[1]
    embed_len_out = x_train_out.shape[1]
    embed_size = 1
    learning_rate = 0.0005
    hidden_size = 64
    epoches = 20
    iterations = 3000
    n_classes = 2
    y_train = one_hot(y_train, n_classes)
    y_test = one_hot(y_test, n_classes)
    # x:(num,seq_len,embed_size)
    # y:(num,1)
    # y_train (num,2)
    g1 = tf.Graph()
    sess1 = tf.Session(graph=g1)
    with sess1.as_default():
        with g1.as_default():
            x_in_place = tf.placeholder(tf.float32,
                                        [None, embed_size * embed_len_in])
            x_in_place = tf.reshape(x_in_place, [-1, embed_len_in, embed_size])
            x_out_place = tf.placeholder(tf.float32,
                                         [None, embed_size * embed_len_out])
            x_out_place = tf.reshape(x_out_place,
                                     [-1, embed_len_out, embed_size])
            mask_in_place = tf.placeholder(tf.int32, shape=None)
            mask_out_place = tf.placeholder(tf.int32, shape=None)
            y_place = tf.placeholder(tf.int32, [None, n_classes])
            # mask_place = tf.placeholder(tf.float32,[None,embed_size])
            output_in, output_out = model_1(x_in_place, x_out_place,
                                            hidden_size, mask_in_place,
                                            mask_out_place)
            co_in_result, co_out_result = model_2(output_in, output_out,
                                                  hidden_size)
            outputs_final_in, outputs_final_out = model_3(
                co_in_result, co_out_result, hidden_size)
            # output_final = tf.concat([outputs_final_in, outputs_final_out], -1)
            output_final = outputs_final_out
            output = tf.layers.dense(inputs=output_final[:, -1, :],
                                     units=n_classes)
            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_place, logits=output)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('adam_optimizer'):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(cross_entropy)
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_place, axis=1),
                                              tf.argmax(output, axis=1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
                prediction = tf.argmax(output, axis=1)
                actual = tf.argmax(y_place, axis=1)
                ones_like_actual = tf.ones_like(actual)
                zeros_like_actual = tf.zeros_like(actual)
                ones_like_prediction = tf.ones_like(prediction)
                zeros_like_prediction = tf.zeros_like(prediction)
                tp_op = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(actual, ones_like_actual),
                            tf.equal(prediction, ones_like_prediction)),
                        'float'))
                tn_op = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(actual, zeros_like_actual),
                            tf.equal(prediction, zeros_like_prediction),
                        ), 'float'))
                fp_op = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(actual, zeros_like_actual),
                            tf.equal(prediction, ones_like_prediction)),
                        'float'))
                fn_op = tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            tf.equal(actual, ones_like_actual),
                            tf.equal(prediction, zeros_like_prediction)),
                        'float'))
            init = tf.global_variables_initializer()
        sess1.run(init)
    index = 0
    X_test_in = x_test_in
    X_test_out = x_test_out
    Y_test = y_test
    for epoch in range(epoches):
        index = 0
        for step in range(iterations):
            X_train_in = x_train_in[index:index + batch_size, :, :]
            X_train_out = x_train_out[index:index + batch_size, :, :]
            Y_train = y_train[index:index + batch_size, :]
            len_in = seq_train_in[index:index + batch_size]
            len_out = seq_train_out[index:index + batch_size]
            index += batch_size
            loss_, output_, op_ = sess1.run(
                [cross_entropy, output, optimizer],
                feed_dict={
                    x_in_place: X_train_in,
                    x_out_place: X_train_out,
                    y_place: Y_train,
                    mask_in_place: len_in,
                    mask_out_place: len_out
                })
            if step % 100 == 0:
                accu_test, tp, fp, tn, fn, loss_test = sess1.run(
                    [accuracy, tp_op, fp_op, tn_op, fn_op, cross_entropy],
                    feed_dict={
                        x_in_place: X_test_in,
                        x_out_place: X_test_out,
                        y_place: Y_test,
                        mask_in_place: seq_test_in,
                        mask_out_place: seq_test_out
                    })
                print(tp, fp, tn, fn)
                if tn == 0:
                    recall, precision = 0, 0
                else:
                    recall = float(tn) / (float(fp) + float(tn))
                    precision = float(tn) / (float(tn) + float(fn))
                # print(tp,fp,tn,fn)
                print(epoch, step, accu_test, recall, precision, loss_,
                      loss_test)

    return 0


if __name__ == '__main__':
    main()
