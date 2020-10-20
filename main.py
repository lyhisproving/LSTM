import tensorflow as tf
import numpy as np


def one_hot(label, num_class):
    label = np.array(label)
    num_label = label.shape[0]
    index = np.arange(num_label) * num_class
    out = np.zeros((num_label, num_class))
    out.flat[index + label.ravel()] = 1
    return out


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
        for x in X
    ])


def load_data(train_file, test_file, train_label, test_label):
    #params: train_file,test_file
    #return X，Y的ndarray数组
    #x_train:(20000,数组最大长度,1)
    #x_test:(10000,数组最大长度,1)
    #y_train:(20000,1)
    #y_test:(10000,1)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    with open(train_file, 'r') as f:
        line = f.readline()
        while line:
            lst = line.strip().split()
            data = [int(x) for x in lst]
            x_train.append(data)
            line = f.readline()

    with open(test_file, 'r') as f:
        line = f.readline()
        while line:
            lst = line.strip().split()
            data = [int(x) for x in lst]
            x_test.append(data)
            line = f.readline()
    with open(train_label, 'r') as f:
        line = f.readline()
        while line:
            lst = line.split()
            data = [int(x) for x in lst]
            y_train.append(data)
            line = f.readline()
    with open(test_label, 'r') as f:
        line = f.readline()
        while line:
            lst = line.split()
            data = [int(x) for x in lst]
            y_test.append(data)
            line = f.readline()

    X = x_train + x_test
    X = seq_padding(X)
    x_train = X[0:len(x_train)]
    x_test = X[len(x_train):]
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    print(np.shape(x_train))
    print(np.shape(x_test))
    print(np.shape(y_train))
    print(np.shape(y_test))
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_data('data/TrainSamples.txt',
                                             'data/TestSamples.txt',
                                             'data/TrainLabel.txt',
                                             'data/TestLabel.txt')

batch_size = 256
time_step = x_train.shape[1]
input_size = 1
learning_rate = 0.005
hidden_size = 128
iterations = 5000
n_classes = 2
y_train = one_hot(y_train, n_classes)
y_test = one_hot(y_test, n_classes)

print(np.shape(y_train))
print(type(y_train))

x_place = tf.placeholder(tf.float32, [None, input_size * time_step])
x_place = tf.reshape(x_place, [-1, time_step, input_size])
y_place = tf.placeholder(tf.int32, [None, n_classes])
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
outputs, states = tf.nn.dynamic_rnn(
    cell=rnn_cell,
    inputs=x_place,
    time_major=False,
    dtype=tf.float32,
)
output = tf.layers.dense(inputs=outputs[:, -1, :],
                         units=n_classes,
                         activation=tf.nn.sigmoid)
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_place, logits=output)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.equal(tf.argmax(y_place, axis=1), tf.argmax(output, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction, 'float'))
prediction = tf.argmax(output, axis=1)
actual = tf.argmax(y_place, axis=1)
ones_like_actual = tf.ones_like(actual)
zeros_like_actual = tf.zeros_like(actual)
ones_like_prediction = tf.ones_like(prediction)
zeros_like_prediction = tf.zeros_like(prediction)
tp_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(tf.equal(actual, ones_like_actual),
                       tf.equal(prediction, ones_like_prediction)), 'float'))
tn_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(actual, zeros_like_actual),
            tf.equal(prediction, zeros_like_prediction),
        ), 'float'))
fp_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(tf.equal(actual, zeros_like_actual),
                       tf.equal(prediction, ones_like_prediction)), 'float'))
fn_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(tf.equal(actual, ones_like_actual),
                       tf.equal(prediction, zeros_like_prediction)), 'float'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_test = x_test[10000:20000]
y_test = y_test[10000:20000]
for step in range(iterations):
    index = 0
    x = x_train[index:index + batch_size, :, :]
    y = y_train[index:index + batch_size, :]
    index += batch_size
    _, loss_train, accuracy_train = sess.run([optimizer, loss, accuracy], {
        x_place: x,
        y_place: y
    })

    if step % 100 == 0:
        #print(out)
        loss_test, accuracy_test, tp, tn, fp, fn = sess.run(
            [loss, accuracy, tp_op, tn_op, fp_op, fn_op], {
                x_place: x_test,
                y_place: y_test
            })
        tpr = float(tp) / (float(tp) + float(fn))
        fpr = float(fp) / (float(fp) + float(tn))
        fnr = float(fn) / (float(tp) + float(fn))
        recall = tpr
        precision = float(tp)/(float(tp) + float(fp))
        print('train loss: %f' % loss_train, '\ttest loss: %f' % loss_test,
              '\ttrain accuracy:%f' % accuracy_train,
              '\ttest accuracy:%f' % accuracy_test,
              '\trecall:%f' % recall, '\tprecision:%f' % precision,
              '\tstep: %d' % step)
        #print('success')
