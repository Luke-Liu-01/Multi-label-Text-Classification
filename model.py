import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()

        # model parameters
        self.mode = args[0]
        self.vocab_num = args[1]
        self.label_num = args[2]
        self.embedding_dim = args[3]
        self.vectors = args[4]
        self.kernel_size = [2, 3, 4]
        self.kernel_num = 100  # eqaul to the number of output channels

        # NN structure
        self.embedding = nn.Embedding(self.vocab_num, self.embedding_dim)
        if self.mode == 'static':
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=True)
        elif self.mode == 'not-static':
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=False)  # fine-tune during training

        # use multiple filters (with varying window sizes) to obtain multiple feature maps
        self.convs = nn.ModuleList()
        for _size in self.kernel_size:
            self.convs.append(
                nn.Conv1d(
                    in_channels=self.embedding_dim,
                    out_channels=self.kernel_num,
                    kernel_size=_size)
            )

        self.dropout = nn.Dropout(p=0.5)

        # 3 * 100 feature vectirs in total (3 different sizes of kernels, 100 filters for each size)
        self.linear = nn.Linear(len(self.kernel_size) * self.kernel_num, self.label_num)

    def forward(self, sentence):
        # sentence: (batch_size, max_len)
        embedding = self.embedding(sentence)  # embedding: (batch_size, max_len, embedding_dim)

        # shape of the kernel: kernel_size * in_channels
        # we need to apply convolutional operation on the last dimension
        embedding = embedding.permute(0, 2, 1)  # adjust the ordering of dimensions to (batch_size, embedding_dim, max_len)

        # feature_maps: (batch_size, output_channel, max_len-kernel_size+1)
        feature_maps = [F.relu(conv(embedding)) for conv in self.convs]

        # apply max-pooling to each feature map and get univariate maps (batch_size, output_channel, 1)
        univariate_vecs = [F.max_pool1d(input=feature_map, kernel_size=feature_map.shape[2]) for feature_map in
                           feature_maps]

        # concatenat these features together to form the final feature vector (batch_size, 3*output_channel, 1)
        univariate_vecs = torch.cat(univariate_vecs, dim=1)
        feature_vec = univariate_vecs.view(-1, univariate_vecs.shape[1])  # -> (batch_size, 3*output_channel)

        # dropout
        feature_vec = self.dropout(feature_vec)

        # fully connected output layer (batch_size, 3*output_channel) -> (3*output_channel, label_num)
        output = self.linear(feature_vec)
        return output


class TextRCNN(nn.Module):
    def __init__(self, args):
        super(TextRCNN, self).__init__()

        self.mode = args[0]
        self.vocab_num = args[1] 
        self.label_num = args[2]
        self.embedding_dim = args[3]
        self.vectors = args[4]
        self.hidden_size = 256  # the number of features in the hidden state h
        self.num_layers = 1  # number of recurrent layers

        # structure
        self.embedding = nn.Embedding(self.vocab_num, self.embedding_dim)
        if self.mode == 'static':
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=True)
        elif self.mode == 'not-static':
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=False)  # fine-tune
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )

        self.linear = nn.Linear(self.embedding_dim + self.hidden_size * 2, self.label_num)

    def forward(self, sentence):
        # sentence: (batch_size, max_len)
        embedding = self.embedding(sentence)  # embedding: (batch_size, max_len, embedding_dim)

        # bi-lstm
        context_info, (c, h) = self.lstm(embedding)  # context_info: (batch_size, max_len, hidden_size*2)

        # represent a word with the combination of the word itself and its context
        # (batch_size, max_len, hidden_size*2+embedding_dim)
        context_info_chunks = torch.chunk(context_info, chunks=2, dim=2)  # split the tensor into left&right contexts
        context_left = context_info_chunks[0]  # left context
        context_right = context_info_chunks[1]  # right context
        representation = torch.cat((context_left, embedding), 2)
        representation = torch.cat((representation, context_right), 2)  # [cl,w,cr]
        representation = F.tanh(representation)

        # ->(batch_size, hidden_size*2+embedding_dim, max_len)
        representation = representation.permute(0, 2, 1)

        # apply max_pooling operation to get feature vectors
        # feature_vec: (batch_size, hidden_size*2+embedding_dim, 1)
        feature_vec = F.max_pool1d(input=representation, kernel_size=representation.shape[-1])
        feature_vec = feature_vec.squeeze(-1)  # ... -> (batch_size, hidden_size*2+embedding_dim)

        output = self.linear(feature_vec)  # (batch_size, hidden_size*2+embedding_dim) -> (batch_size, label_num)

        return output


# replace the max-pooling in RCNN with soft-attention
class TextRANN(nn.Module):
    def __init__(self, args):
        super(TextRANN, self).__init__()

        self.mode = args[0]
        self.vocab_num = args[1]
        self.label_num = args[2]
        self.embedding_dim = args[3]
        self.vectors = args[4]
        self.hidden_size = 256
        self.num_layers = 1

        # structure
        self.embedding = nn.Embedding(self.vocab_num, self.embedding_dim)
        if self.mode == 'static':
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=True)
        elif self.mode == 'not-static':
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=False)  # fine-tune
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )

        self.linear = nn.Linear(self.embedding_dim + self.hidden_size * 2, self.label_num)

    def forward(self, sentence):
        # sentence: (batch_size, max_len)
        embedding = self.embedding(sentence)  # embedding: (batch_size, max_len, embedding_dim)

        # bi-lstm
        context_info, (c, h) = self.lstm(embedding)  # context_info: (batch_size, max_len, hidden_size*2)

        # represent a word with the combination of the word itself and its context
        # (batch_size, max_len, hidden_size*2+embedding_dim)
        context_info_chunks = torch.chunk(context_info, chunks=2, dim=2)  # split the tensor into left&right contexts
        context_left = context_info_chunks[0]  # left context
        context_right = context_info_chunks[1]  # right context
        representation = torch.cat((context_left, embedding), 2)
        representation = torch.cat((representation, context_right), 2)  # [cl,w,cr]
        key = F.tanh(representation)

        # attention
        alpha = F.softmax(torch.matmul(key, self.query), dim=1).unsqueeze(-1)  # (batch_size, max_len, 1)
        feature_vec = representation * alpha  # (batch_size, max_len, hidden_size*2+embedding_dim)
        feature_vec = F.relu(torch.sum(feature_vec, 1))  # (batch_size, hidden_size*2+embedding_dim)

        output = self.linear(feature_vec)  # (batch_size, hidden_size*2+embedding_dim) -> (batch_size, label_num)

        return output
