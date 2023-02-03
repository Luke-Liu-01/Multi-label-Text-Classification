import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MultiLabelDataSet
from dataset import Tokenizer
from model import TextCNN
import pandas as pd
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# save the predicted results
def SavaPrediction():
    print('load data...')
    data_set = MultiLabelDataSet('static')
    print('load model...')
    model = torch.load('models/textcnn_multi_label.pth')

    train_iter, dev_iter = data_set.GetIter()

    with open('data/data_predicted.txt', 'w', encoding='utf-8') as f:
        for batch in dev_iter:
            x = batch.text
            x = x.permute(1, 0).to(DEVICE)  # (max_len, batch_size) -> (batch_size, max_len)
            output = model(x)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            result_list = output.int().cpu().detach().numpy().tolist()
            for result in result_list:
                result_str = list(map(str, result))
                label = '-'.join(result_str)
                f.write(label)
                f.write('\n')


# user test
def Test(sentence):

    print('load data...')
    data_set = MultiLabelDataSet('static')
    print('load model...')
    textcnn = torch.load('models/textcnn_multi_label_80.pth')
    textrcnn = torch.load('models/textrcnn_multi_label_80.pth')

    # args = data_set.GetArgs()
    # model = TextCNN(args)
    # model.load_state_dict(torch.load('models/textcnn_params_multi_label.pth'))

    token = Tokenizer(sentence)
    indices = data_set.text.vocab.lookup_indices(token)  # str -> index
    for i in range(5):  # padding
        indices.append(0)
    x = torch.Tensor(indices).to(torch.int64)
    x = x.unsqueeze(0).to(DEVICE)  # (len) -> (1, len)

    output_cnn = textcnn(x)
    output_cnn[output_cnn >= 0.5] = 1
    output_cnn[output_cnn < 0.5] = 0

    output_rcnn = textrcnn(x)
    output_rcnn[output_rcnn >= 0.5] = 1
    output_rcnn[output_rcnn < 0.5] = 0

    max_value_index = torch.max(output_cnn+output_rcnn, 1)[1]
    output = output_cnn + output_rcnn  # take two models' results into account
    output[output >= 1] = 1
    output[output < 1] = 0

    # if all the probs are smaller than the threshold, output the label with max prob
    for i in range(output.shape[0]):
        output[i][max_value_index[i]] = 1  
    result_list = output.int().cpu().detach().numpy().tolist()[0]
    label_list = ['空间', '动力', '操控', '能耗', '舒适性', '外观', '内饰', '性价比', '配置', '续航', '安全性',
                  '环保', '质量与可靠性', '充电', '服务', '品牌', '智能驾驶', '其它']
    labels_predicted = []
    for index, value in enumerate(result_list):
        if value == 1:
            labels_predicted.append(label_list[index])
    print(labels_predicted)


# count co-coccurring patterns
def CoOccurring():
    df = pd.read_csv('data/multi_label_train.csv')
    labels = df['label']
    co_occur_lables = {}
    for i in range(len(labels)):
        label = labels[i].split('-')
        label_pattern = []
        for index, value in enumerate(label):
            if value == '1':  # ignore single-label data
                label_pattern.append(str(index))
        if len(label_pattern) > 1:
            label_pattern = ' '.join(label_pattern)
            if label_pattern not in co_occur_lables.keys():
                co_occur_lables.update({label_pattern: 1})
            else:
                co_occur_lables[label_pattern] += 1
    co_occur_order = sorted(co_occur_lables.items(), key=lambda x: x[1], reverse=True)
    with open('data/co_occurring_labels.txt', 'w', encoding='utf-8') as f:
        for i in range(len(co_occur_order)):
            pattern = co_occur_order[i][0]
            count = co_occur_order[i][1]
            f.write(pattern + ' ' + str(count) + '\n')


if __name__ == '__main__':
    # SavaPrediction()
    Test('空间很大，比较舒服。')