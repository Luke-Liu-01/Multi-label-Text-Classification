import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MultiLabelDataSet
from model import *
import numpy as np

EPOCHS = 80
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('{} environment...'.format(DEVICE))


# DEVICE = torch.device('cpu')


def Accuracy(output, y):
    output = torch.sigmoid(output)
    max_value_index = torch.max(output, 1)[1]
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    # if all the probs are smaller than the threshold, output the label with max prob
    for i in range(output.shape[0]):
        output[i][max_value_index[i]] = 1

    result = output.long() + y.long()
    result = result.reshape(1, -1)
    result_list = result.cpu().numpy().tolist()[0]
    correct = result_list.count(2)  # 2 means correctly match
    total = result_list.count(1) + correct
    return correct / total


def MultiLabelTrain(data_set, model):
    train_iter, dev_iter = data_set.GetIter()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        batch_acc = 0
        batch_num = 0
        loss_per_epoch = 0.0
        for batch in train_iter:
            x = batch.text
            y = batch.label.to(DEVICE)
            x = x.permute(1, 0).to(DEVICE)  # (max_len, batch_size) -> (batch_size, max_len)
            optimizer.zero_grad()
            output = model(x)
            loss_func = nn.BCEWithLogitsLoss()  # BCE loss is more tailored and suitable for multi-label problems
            y = y.permute(1, 0)  # label_num * batch_size -> batch_size * label_num
            loss = loss_func(output, y.float())
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.item()
            batch_acc += Accuracy(output, y)
            batch_num += 1

        acc_train = batch_acc / batch_num
        acc_valid = MultiLabelEvaluate(dev_iter, model)
        print('Epoch {} - Loss: {}  train_acc: {}  valid_acc: {}'.format(epoch, loss_per_epoch, acc_train, acc_valid))


def MultiLabelEvaluate(dev_iter, model):
    batch_acc = 0
    batch_num = 0
    model.eval().to(DEVICE)
    for batch in dev_iter:
        x = batch.text
        y = batch.label.to(DEVICE)
        x = x.permute(1, 0).to(DEVICE)  # (max_len, batch_size) -> (batch_size, max_len)
        output = model(x)
        y = y.permute(1, 0)

        batch_acc += Accuracy(output, y)
        batch_num += 1
    acc_rate = batch_acc / batch_num
    return acc_rate


# label co-occurrence initialization methods
def SetWeight(model, mode):
    if mode == 'TextRCNN':
        label_num = model.label_num
        embedding_dim = model.embedding_dim
        hidden_size = model.hidden_size

        # default initialization parameters
        bound = 1 / np.sqrt(hidden_size * 2 + embedding_dim)  # sqrt(1/the dimension of input)
        weight = np.random.uniform(-bound, bound, size=(hidden_size * 2 + embedding_dim, label_num))
        hidden_output_num = hidden_size * 2 + embedding_dim + label_num  # number of hidden and output units
    elif mode == 'TextCNN':
        label_num = model.label_num
        kernel_size = model.kernel_size
        kernel_num = model.kernel_num

        # default initialization parameters
        bound = 1 / np.sqrt(len(kernel_size) * kernel_num)
        weight = np.random.uniform(-bound, bound, size=(len(kernel_size) * kernel_num, label_num))
        hidden_output_num = len(kernel_size) * kernel_num + label_num 

    label_pattern = []  # co-occurring label patterns
    pattern_num = 50  # only choose the top 50 frequent patterns
    with open('data/co_occurring_labels.txt', 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            if index > pattern_num:
                break
            labels = list(map(int, line.strip().split(' ')))  #[label-1 ... label-n count]
            label_pattern.append(labels)

    # modify the weights
    for i in range(pattern_num):
        count = label_pattern[i][-1]
        pattern = label_pattern[i][0:-1]
        c = np.sqrt(count) * np.sqrt(6) / np.sqrt(hidden_output_num)  # the weight C
        # For each pattern corresponding to the co-occurring labels, 
        # the row is initialized in the way that 
        # the weight of elements corresponding to the co-occurring labels is a constant C 
        # and the other elements are assigned a value of 0
        weight_row = np.zeros(label_num)
        for index in pattern:
            weight_row[index] = c
        weight[i] = weight_row

    model.linear.weight = torch.nn.Parameter(torch.Tensor(weight.T))
    return model


def main(mode, set_weight=False):
    print('load data...')
    data_set = MultiLabelDataSet('static')

    print('build model...')
    args = data_set.GetArgs()

    if mode == 'TextCNN':
        model = TextCNN(args).to(DEVICE)
    elif mode == 'TextRCNN':
        model = TextRCNN(args).to(DEVICE)
    elif mode == 'TextRANN':
        model = TextRANN(args).to(DEVICE)

    if set_weight:
        print('set weight...')
        model = SetWeight(model, mode).to(DEVICE)  # label co-occurrence initialization methods
    print('training...')
    MultiLabelTrain(data_set, model)
    print('save model...')
    torch.save(model.state_dict(), 'models/{}_params_multi_label_{}.pth'.format(mode.lower(), EPOCHS))
    torch.save(model, 'models/{}_multi_label_{}.pth'.format(mode.lower(), EPOCHS))


if __name__ == '__main__':
    model_name = 'TextRCNN'
    set_weight = False  # whether to use label co-occurrence initialization methods
    main(model_name, set_weight)
