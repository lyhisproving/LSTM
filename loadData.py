import numpy as np


def normalization(data, max_num, min_num):
    out = []
    for line in data:
        temp = [(x - min_num) / (max_num - min_num) for x in line]
        out.append(temp)
    # x = y *100
    return out


def padding_array(X, max_len, padding=0):
    return np.array([
        np.concatenate([x, [padding] *
                        (max_len - len(x))]) if len(x) < max_len else x
        for x in X
    ])


def embeding(data, embed_size):
    result = []
    for X in data:
        tmp = np.array([
            np.concatenate([x, [0] * (embed_size - len(x))])
            if len(x) < embed_size else x for x in X
        ])
        result.append(tmp)
    return result


def padding_sequence(X, max_len, embed_size, padding=0):
    return np.array([
        np.concatenate([x, [[padding] * embed_size] *
                        (max_len - len(x))]) if len(x) < max_len else x
        for x in X
    ])


def load_array(train_file, test_file, train_label, test_label):
    # params: train_file,test_file
    # return X，Y的ndarray数组
    # x_train:(20000,embed_len_in,embed_size)
    # x_test:(10000,embed_len_in,embed_size)
    # y_train:(20000,1)
    # y_test:(10000,1)
    # return:
    # X,Y training set and test set
    # mask: the true length of sequence
    x_train_in = []
    x_train_out = []
    x_test_in = []
    x_test_out = []
    y_train = []
    y_test = []
    max_num = -1e5
    min_num = 1e5
    embed_size = 1
    with open(train_file, 'r') as f:
        line = f.readline()
        while line:
            lt = line.split('#')
            lt_1 = lt[0].split()
            lt_2 = lt[1].split()
            data_in = [int(x) for x in lt_1]
            data_out = [int(x) for x in lt_2]
            max_num = max(max_num, max(max(data_in), max(data_out)))
            min_num = min(min_num, min(min(data_in), min(data_out)))
            x_train_in.append(data_in)
            x_train_out.append(data_out)
            line = f.readline()
    with open(test_file, 'r') as f:
        line = f.readline()
        while line:
            lt = line.split('#')
            lt_1 = lt[0].split()
            lt_2 = lt[1].split()
            data_in = [int(x) for x in lt_1]
            data_out = [int(x) for x in lt_2]
            max_num = max(max_num, max(max(data_in), max(data_out)))
            min_num = min(min_num, min(min(data_in), min(data_out)))
            x_test_in.append(data_in)
            x_test_out.append(data_out)
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
    # normalization
    x_train_in = normalization(x_train_in, max_num, min_num)
    x_train_out = normalization(x_train_out, max_num, min_num)
    x_test_in = normalization(x_test_in, max_num, min_num)
    x_test_out = normalization(x_test_out, max_num, min_num)
    # padding
    X_in = x_train_in + x_test_in
    X_out = x_train_out + x_test_out
    L1 = (len(x) for x in X_in)
    max_len_in = max(L1)
    L2 = (len(x) for x in X_out)
    max_len_out = max(L2)
    X_in = padding_array(X_in, max_len_in)
    X_out = padding_array(X_out, max_len_out)
    # mask
    seq_len_in = [len(x) for x in X_in]
    seq_len_out = [len(x) for x in X_out]

    X_in = np.array(X_in)
    X_out = np.array(X_out)
    x_train_in = X_in[0:len(x_train_in)]
    x_test_in = X_in[len(x_train_in):]
    x_train_out = X_out[0:len(x_train_out)]
    x_test_out = X_out[len(x_train_out):]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    seq_len_in = np.array(seq_len_in)
    seq_len_out = np.array(seq_len_out)
    x_train_in = x_train_in.reshape(x_train_in.shape[0], x_train_in.shape[1],
                                    embed_size)
    x_test_in = x_test_in.reshape(x_test_in.shape[0], x_test_in.shape[1],
                                  embed_size)
    x_train_out = x_train_out.reshape(x_train_out.shape[0],
                                      x_train_out.shape[1], embed_size)
    x_test_out = x_test_out.reshape(x_test_out.shape[0], x_test_out.shape[1],
                                    embed_size)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    # print(np.shape(x_train),np.shape(x_test))
    return x_train_in, x_train_out, x_test_in, x_test_out, y_train, y_test, seq_len_in, seq_len_out


def load_sequence(train_file, test_file, train_label, test_label):
    x_train_in = []
    x_train_out = []
    x_test_in = []
    x_test_out = []
    y_train = []
    y_test = []
    max_num = -1e5
    min_num = 1e5
    embed_size = 10
    with open(train_file, 'r') as f:
        line = f.readline()
        while line:
            sentence_in = []
            sentence_out = []
            lt = line.split('#')
            lt_1 = lt[0].split()
            lt_2 = lt[1].split()
            for word in lt_1:
                charactors = list(word)
                arr_in = [ord(ch) for ch in charactors]
                if (len(charactors) < embed_size):
                    arr_in = arr_in + [0] * (embed_size - len(charactors))
                max_num = max(max_num, max(arr_in))
                min_num = min(min_num, min(arr_in))
                sentence_in.append(arr_in)
            for word in lt_2:
                charactors = list(word)
                arr_out = [ord(ch) for ch in charactors]
                if (len(charactors) < embed_size):
                    arr_out = arr_out + [0] * (embed_size - len(charactors))
                max_num = max(max_num, max(arr_out))
                min_num = min(min_num, min(arr_out))
                sentence_out.append(arr_out)
            x_train_in.append(sentence_in)
            x_train_out.append(sentence_out)
            
            line = f.readline()
    with open(test_file, 'r') as f:
        line = f.readline()
        while line:
            sentence_in = []
            sentence_out = []
            lt = line.split('#')
            lt_1 = lt[0].split()
            lt_2 = lt[1].split()
            for word in lt_1:
                charactors = list(word)
                arr_in = [ord(ch) for ch in charactors]
                if (len(charactors) < embed_size):
                    arr_in = arr_in + [0] * (embed_size - len(charactors))
                max_num = max(max_num, max(arr_in))
                min_num = min(min_num, min(arr_in))
                sentence_in.append(arr_in)
            for word in lt_2:
                charactors = list(word)
                arr_out = [ord(ch) for ch in charactors]
                if (len(charactors) < embed_size):
                    arr_out = arr_out + [0] * (embed_size - len(charactors))
                max_num = max(max_num, max(arr_out))
                min_num = min(min_num, min(arr_out))
                sentence_out.append(arr_out)
            x_test_in.append(sentence_in)
            x_test_out.append(sentence_out)
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
    # padding
    #print(np.shape(x_train_in),np.shape(x_train_out))
    X_in = x_train_in + x_test_in
    X_out = x_train_out + x_test_out
    L1 = (len(x) for x in X_in)
    max_len_in = max(L1)
    L2 = (len(x) for x in X_out)
    max_len_out = max(L2)
    X_in = padding_sequence(X_in, max_len_in, embed_size=embed_size)
    X_out = padding_sequence(X_out, max_len_out, embed_size=embed_size)
    # mask
    
    seq_len_in = [len(x) for x in X_in]
    seq_len_out = [len(x) for x in X_out]

    X_in = np.array(X_in)
    X_out = np.array(X_out)
    x_train_in = X_in[0:len(x_train_in)]
    x_test_in = X_in[len(x_train_in):]
    x_train_out = X_out[0:len(x_train_out)]
    x_test_out = X_out[len(x_train_out):]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    seq_len_in = np.array(seq_len_in)
    seq_len_out = np.array(seq_len_out)
    x_train_in = x_train_in.reshape(x_train_in.shape[0], x_train_in.shape[1],
                                    -1)
    x_test_in = x_test_in.reshape(x_test_in.shape[0], x_test_in.shape[1], -1)
    x_train_out = x_train_out.reshape(x_train_out.shape[0],
                                      x_train_out.shape[1], -1)
    x_test_out = x_test_out.reshape(x_test_out.shape[0], x_test_out.shape[1],
                                    -1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    return x_train_in, x_train_out, x_test_in, x_test_out, y_train, y_test, seq_len_in, seq_len_out
