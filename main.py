import tensorflow as tf
import numpy as np
<<<<<<< HEAD

=======
from torch import nn,Tensor
import torch
import torch.nn.functional as F
>>>>>>> a788321a6272f1e8a742c9526cc903686bb4a39c
def one_hot(label, num_class):
    label = np.array(label)
    num_label = label.shape[0]
    index = np.arange(num_label) * num_class
    out = np.zeros((num_label, num_class))
    out.flat[index + label.ravel()] = 1
    return out

<<<<<<< HEAD

=======
>>>>>>> a788321a6272f1e8a742c9526cc903686bb4a39c
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
        for x in X
    ])

<<<<<<< HEAD

def load_data(train_file, test_file, train_label, test_label):
    #params: train_file,test_file
    #return X，Y的ndarray数组
    #x_train:(20000,time_step,embed_size)
    #x_test:(10000,time_step,embed_size)
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
=======
def load_data(train_file, test_file, train_label, test_label):
    #params: train_file,test_file
    #return X，Y的ndarray数组
    #x_train:(20000,seq_len_in,embed_size)
    #x_test:(10000,seq_len_in,embed_size)
    #y_train:(20000,1)
    #y_test:(10000,1)
    x_train_in = []
    x_train_out = []
    x_test_in = []
    x_test_out = []
    y_train = []
    y_test = []
    max_int_seq_size=0
    max_out_seq_size=0
    with open(train_file, 'r') as f:
        line = f.readline()
        while line:
            lt = line.split('#')
            lt_1 = lt[0].split()
            lt_2 = lt[1].split()
            max_int_seq_size=max(max_int_seq_size,len(lt_1))
            max_out_seq_size=max(max_out_seq_size,len(lt_2))
            data_in = [int(x) for x in lt_1]
            data_out=[int(x) for x in lt_2]
            x_train_in.append(data_in)
            x_train_out.append(data_out)
            #lst = lt_1+lt_2
            #data = [int(x) for x in lst]
            #x_train.append(data)
>>>>>>> a788321a6272f1e8a742c9526cc903686bb4a39c
            line = f.readline()

    with open(test_file, 'r') as f:
        line = f.readline()
        while line:
<<<<<<< HEAD
            lst = line.strip().split()
            data = [int(x) for x in lst]
            x_test.append(data)
=======
            lt = line.split('#')
            lt_1 = lt[0].split()
            lt_2 = lt[1].split()
            max_int_seq_size=max(max_int_seq_size,len(lt_1))
            max_out_seq_size=max(max_out_seq_size,len(lt_2))
            data_in = [int(x) for x in lt_1]
            data_out = [int(x) for x in lt_2]
            x_test_in.append(data_in)
            x_test_out.append(data_out)

            #lst = lt_1+lt_2
            #data = [int(x) for x in lst]
            #x_test.append(data)
>>>>>>> a788321a6272f1e8a742c9526cc903686bb4a39c
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
<<<<<<< HEAD

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
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_data('data/TrainSamples.txt',
                                             'data/TestSamples.txt',
                                             'data/TrainLabel.txt',
                                             'data/TestLabel.txt')

batch_size = 256
time_step = x_train.shape[1]
embed_size = 1
learning_rate = 0.005
hidden_size = 64
iterations = 5000
n_classes = 2
y_train = one_hot(y_train, n_classes)  #(200000,2)
y_test = one_hot(y_test, n_classes)  #(40000,2)

x_place = tf.placeholder(tf.float32, [None, embed_size * time_step])
x_place = tf.reshape(x_place, [-1, time_step, embed_size])
y_place = tf.placeholder(tf.int32, [None, n_classes])
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
#output: (batch_size,time_step,n_units) state: (2,batch_size,n_units)
outputs_first, states_first = tf.nn.dynamic_rnn(
    cell=rnn_cell,
    inputs=x_place,
    time_major=False,
    dtype=tf.float32,
)
matrix = tf.nn.linear
output = tf.layers.dense(inputs=outputs_first[:, -1, :],
=======
    
    X_in = x_train_in+x_test_in
    X_out=x_train_out+x_test_out
    X_in=seq_padding(X_in)
    X_out = seq_padding(X_out)
    X_in=np.array(X_in)
    X_out=np.array(X_out)
    x_train_in=X_in[0:len(x_train_in)]
    x_test_in=X_in[len(x_train_in):]
    x_train_out=X_out[0:len(x_train_out)]
    x_test_out=X_out[len(x_train_out):]

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_train_in = x_train_in.reshape(x_train_in.shape[0], x_train_in.shape[1], 1)
    x_test_in = x_test_in.reshape(x_test_in.shape[0], x_test_in.shape[1], 1)
    x_train_out = x_train_out.reshape(x_train_out.shape[0],x_train_out.shape[1],1)
    x_test_out = x_test_out.reshape(x_test_out.shape[0],x_test_out.shape[1],1)

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    #print(np.shape(x_train),np.shape(x_test))
    return x_train_in, x_train_out, x_test_in,x_test_out,y_train, y_test


x_train_in, x_train_out, x_test_in,x_test_out,y_train,y_test = load_data('data/TrainSamples_10_26.txt',
                                             'data/TestSamples_10_26.txt',
                                             'data/TrainLabel_10_26.txt',
                                             'data/TestLabel_10_26.txt')




def model_1(x_in,x_out,hidden_size):
    with tf.name_scope('LSTM_1'):
        cell_in = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
        outputs_in,states_in = tf.nn.dynamic_rnn(
            cell = cell_in,
            inputs = x_in,
            time_major = False,
            dtype = tf.float32
        )
    with tf.name_scope('LSTM_2'):
        cell_out = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
        outputs_out,states_out = tf.nn.dynamic_rnn(
            cell=cell_out,
            inputs = x_out,
            time_major = False,
            dtype = tf.float32
        )
    return outputs_in,outputs_out

def model_2(outputs_in,outputs_out,hidden_size):
    with tf.name_scope('coAttention'):
        dense = tf.layers.Dense(hidden_size,activation='relu')
        x = dense(outputs_in)
        matrix = tf.matmul(x,tf.transpose(outputs_out,perm=[0,2,1]))
        input_matrix = mask_place.unsqueeze(1)
        output_matrix = mask_place.unsqueeze(-1)
        input_weight = tf.nn.softmax(input_matrix,dim=1)
        output_weight = tf.nn.softmax(output_matrix,dim=-1)
        #(batch_size,seq_len,hidden_size)
        coat_in = tf.matmul(input_weight,outputs_out)
        coat_out = tf.matmul(tf.transpose(output_weight,perm=[0,2,1]),outputs_in)
        co_in_result = tf.concat((coat_in,outputs_in),-1)
        co_out_result = tf.concat((coat_out,outputs_out),-1)
    return co_in_result,co_out_result

def model_3(co_in_result,co_out_result,hidden_size):
    with tf.name_scope('LSTM_3'):
        cell_in = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
        output_final_in,state_final_in = tf.nn.dynamic_rnn(
            cell = cell_in,
            inputs = co_in_result,
            time_major = False,
            dtype = tf.float32
        )
    with tf.name_scope('LSTM_4'):
        cell_out = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
        output_final_out,state_final_out = tf.nn.dynamic_rnn(
            cell=cell_out,
            inputs = co_out_result,
            time_major = False,
            dtype = tf.float32
        )
    return output_final_in,output_final_out

    
def main():
    batch_size = 64
    seq_len_in = x_train_in.shape[1]
    seq_len_out = x_train_out.shape[1]
    print(type(seq_len_in),seq_len_in)
    print(type(seq_len_out),seq_len_out)
    embed_size = 1
    learning_rate = 0.005
    hidden_size = 64
    iterations = 5000
    n_classes = 2
    y_train = one_hot(y_train, n_classes)  #(200000,2)
    y_test = one_hot(y_test, n_classes)  #(40000,2)
    # x:(num,seq_len,embed_size)
    # y:(num,1)
    # y_train (num,2)
    g1 = tf.Graph()
    sess1 = tf.Session(graph=g1)
    with sess1.as_default():
        with g1.as_default():
            x_in_place = tf.placeholder(tf.float32, [None, embed_size * seq_len_in])
            x_in_place = tf.reshape(x_in_place, [-1, seq_len_in, embed_size])
            x_out_place = tf.placeholder(tf.float32,[None,embed_size* seq_len_out])
            x_out_place = tf.reshape(x_out_place,[-1,seq_len_out,embed_size])
            output_in,output_out = model_1(x_in_place,x_out_place,hidden_size)
            co_in_result,co_out_result = model_2(output_in,output_out,hidden_size)
            outputs_final_in,outputs_final_out = model_3(co_in_result,co_out_result,hidden_size)
            output_final = outputs_final_in+outputs_final_out
            output = tf.layers.dense(inputs=output_final[:,-1,:],units=n_classes,activation=tf.nn.softmax)

    for step
    return 1
if __name__ == '__main__':
    main()
'''x_in_place = tf.placeholder(tf.float32, [None, embed_size * seq_len_in])
x_in_place = tf.reshape(x_in_place, [-1, seq_len_in, embed_size])
x_out_place = tf.placeholder(tf.float32,[None,embed_size* seq_len_out])
x_out_place = tf.reshape(x_out_place,[-1,seq_len_out,embed_size])
mask_place = tf.placeholder(tf.float32,[None,embed_size])
y_place = tf.placeholder(tf.int32, [None, n_classes])
embed_cell_in = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
embed_cell_out = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
#output: (batch_size,seq_len_in,n_units_in) state: (2,batch_size,n_units)

outputs_in, states_in = tf.nn.dynamic_rnn(
    cell=embed_cell_in,
    inputs=x_in_place,
    time_major=False,
    dtype=tf.float32,
)
#(batch_size,seq_len_out,n_units_out)
tf.reset_default_graph()
with tf.Graph().as_default():
    outputs_out,states_out = tf.nn.dynamic_rnn(
        cell=embed_cell_out,
        inputs = x_out_place,
        time_major=False,
        dtype=tf.float32,
    )
dense = tf.layers.Dense(hidden_size,activation='relu')
x = dense(outputs_in)
#(batch,seq_len_in,seq_len_out)
matrix = tf.matmul(x,tf.transpose(outputs_out,perm=[0,2,1]))
#这里少了一步mask_fill
#(batch,1,seq_len)
input_matrix = mask_place.unsqueeze(1)
output_matrix = mask_place.unsqueeze(-1)
input_weight = tf.nn.softmax(input_matrix,dim=1)
output_weight = tf.nn.softmax(output_matrix,dim=-1)
#(batch_size,seq_len,hidden_size)
coat_in = tf.matmul(input_weight,outputs_out)
coat_out = tf.matmul(tf.transpose(output_weight,perm=[0,2,1]),outputs_in)
co_in_result = tf.concat((coat_in,outputs_in),-1)
co_out_result = tf.concat((coat_out,outputs_out),-1)
model_cell_in = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
model_cell_out = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
outputs_final_in,states_final_in = tf.nn.dynamic_rnn(
    cell=model_cell_in,
    inputs= co_in_result,
    time_major=False,
    dtype=tf.float32,
)
outputs_final_out,states_final_out = tf.nn.dynamic_rnn(
    cell = model_cell_out,
    inputs = co_out_result,
    time_major = False,
    dtype=tf.float32,
)
outputs_final = outputs_final_in+outputs_final_out
states_final = states_final_in+states_final_out

output = tf.layers.dense(inputs=outputs_final[:, -1, :],
>>>>>>> a788321a6272f1e8a742c9526cc903686bb4a39c
                         units=n_classes,
                         activation=tf.nn.softmax)
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
#x = x_train[0:batch_size, :, :]
#y = y_train[0:batch_size, :]
#sess.run({x_place: x, y_place: y})

<<<<<<< HEAD
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
        loss_test, accuracy_test, tp, tn, fp, fn, output_, state_ = sess.run(
            [
                loss, accuracy, tp_op, tn_op, fp_op, fn_op, outputs_first,
                states_first
            ], {
                x_place: x_test,
                y_place: y_test
            })
        print(np.shape(output_))
=======

for step in range(iterations):
    index = 0
    X_train_in = x_train_in[index:index + batch_size, :, :]
    
    X_train_out = x_train_out[index:index + batch_size, :, :]

    mask_train = y_train[index:index+batch_size,:]
    Y_train = y_train[index:index + batch_size, :]
    index += batch_size
    _, loss_train, accuracy_train = sess.run([optimizer, loss, accuracy], {
        x_in_place: X_train_in,
        x_out_place:X_train_out,
        mask_place:mask_train,
        y_place:Y_train
    })
    if step % 100 == 0:
        #print(out)
        batch = 1000
        X_test_in = x_test_in[index:index+batch,:,:]
        X_test_out = x_test_out[index:index+batch,:,:]
        mask_test = y_test[index:index+batch,:,:]
        Y_test = y_test[index:index+batch,:,:]
        loss_test, accuracy_test, tp, tn, fp, fn, output_, state_ = sess.run(
            [
                loss, accuracy, tp_op, tn_op, fp_op, fn_op, outputs_final,
                states_final
            ], {
                x_in_place:X_test_in,
                x_out_place:X_test_out,
                mask_place:mask_test,
                y_place:Y_test
            })
        print(type(output_))
>>>>>>> a788321a6272f1e8a742c9526cc903686bb4a39c
        print(np.shape(state_))
        tpr = float(tp) / (float(tp) + float(fn))
        fpr = float(fp) / (float(fp) + float(tn))
        fnr = float(fn) / (float(tp) + float(fn))
        recall = tpr
        precision = float(tp) / (float(tp) + float(fp))
        print('train loss: %f' % loss_train, '\ttest loss: %f' % loss_test,
              '\ttrain accuracy:%f' % accuracy_train,
              '\ttest accuracy:%f' % accuracy_test, '\trecall:%f' % recall,
              '\tprecision:%f' % precision, '\tstep: %d' % step)
        #print('success')
<<<<<<< HEAD
=======
'''

>>>>>>> a788321a6272f1e8a742c9526cc903686bb4a39c
