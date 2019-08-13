import numpy as np
import random
import torch
import nltk
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

class RealDataLoader():
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length
        # self.word2index = {'<PAD>': 0, '<SOS>': sentiment.1, '<EOS>': 2, '<unk>': 3}
        # self.index2word = {0: '<PAD>', sentiment.1: '<SOS>', 2: "<EOS>", 3: '<unk>'}
        self.word2index = {}
        self.index2word = {}


    def Vocabulary(self, voc_file):

        n = 0
        with open(voc_file, 'r') as voc:
            for i in voc:
                self.word2index[i.strip('\n')] = n
                n += 1

        self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))
        return self.word2index, self.index2word


    def create_batches(self, neg_file, pos_file, shuff=True):

        data_pos = self.makeData(pos_file, label=1)
        data_neg= self.makeData(neg_file, label=0)

        total_data = self.shuff(data_neg, data_pos, shuff=shuff)
        self.sequence_batches = self.makeBatch(total_data)
        self.num_batch =len(self.sequence_batches)

        self.pointer = 0


    def next_batch(self):
        ret = self.sequence_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def random_batch(self):
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret = self.sequence_batches[rn_pointer]
        return ret

    def reset_pointer(self):
        self.pointer = 0


    def makeData(self, srcFile, label):


        text_keyword = []
        text_keyword_pos = []
        text_keyword_length = []
        text_keyword_id = []
        original_text = []
        original_id = []
        original_label = []
        original_length = []


        print('Processing %s  ...' % srcFile)
        srcF = open(srcFile, "r")
        # srcSet = srcF.readlines()

        keep_word_property = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'PRP', 'PRP$']
        # keep_word_property = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']

        num = 0
        for i in srcF:
            if num%5000 == 0:
                print(num)

            num +=1
            text = i.strip(' \n')
            split_text = text.split()
            # if len(split_text) > 15:
            #     continue

            keyword = []
            keyword_pos = []
            text_id_line = []
            keyword_id_line = []

            text_line = ["<SOS>"] + split_text + ["<END>"]
            original_text.append(text_line)
            original_length.append(len(text_line))
            original_label.append(label)

            pos_list = nltk.pos_tag(split_text)
            for tube in pos_list:
                if tube[1] in keep_word_property:
                    keyword.append(tube[0])
                    keyword_pos.append(tube)

            keyword = ["<START>"] + keyword + ["<END>"]
            text_keyword.append(keyword)
            text_keyword_length.append(len(keyword))
            text_keyword_pos.append(keyword_pos)

            for j in range(len(keyword)):
                try:
                    id = self.word2index[keyword[j]]
                except KeyError:
                    id = 3
                keyword_id_line.append(id)

            for j in range(len(text_line)):
                try:
                    id = self.word2index[text_line[j]]
                except KeyError:
                    id = 3
                text_id_line.append(id)

            text_keyword_id.append(keyword_id_line)
            original_id.append(text_id_line)

        # for i in range(len(srcSet)):
        #     if i % 5000 == 0:
        #         print("now: ", i)
        #
        #     original_label.append(label)
        #     keyword = []
        #     keyword_pos = []
        #     text_id_line = []
        #     keyword_id_line = []
        #
        #
        #     srcSet[i] = srcSet[i].strip('\n').strip()
        #     split_text = srcSet[i].split()
        #
        #     text_line = ["<START>"] + split_text + ["<END>"]
        #     original_text.append(text_line)
        #     # if len(text_line) > self.seq_length:
        #     #     print()
        #     #     continue
        #     original_length.append(len(text_line))
        #
        #     pos_list = nltk.pos_tag(split_text)
        #     for tube in pos_list:
        #         if tube[1] in keep_word_property:
        #             keyword.append(tube[0])
        #             keyword_pos.append(tube)
        #
        #     keyword = ["<START>"] + keyword + ["<END>"]
        #     text_keyword.append(keyword)
        #     text_keyword_length.append(len(keyword))
        #     text_keyword_pos.append(keyword_pos)




            # for j in range(len(text_keyword[i])):
            #     try:
            #         id = self.word2index[text_keyword[i][j]]
            #     except KeyError:
            #         id = 3
            #     keyword_id_line.append(id)
            #
            # for j in range(len(original_text[i])):
            #     try:
            #         id = self.word2index[original_text[i][j]]
            #     except KeyError:
            #         id = 3
            #     text_id_line.append(id)
            #
            # text_keyword_id.append(keyword_id_line)
            # original_id.append(text_id_line)

        print('... padding')
        for i in range(len(original_text)):
            if original_length[i] < self.seq_length:
                for j in range(self.seq_length - original_length[i]):
                    original_text[i].append("<PAD>")
                    original_id[i].append(0)

        for i in range(len(text_keyword)):
            if text_keyword_length[i] < self.seq_length:
                for j in range(self.seq_length - text_keyword_length[i]):
                    text_keyword[i].append("<PAD>")
                    text_keyword_id[i].append(0)

        Dataset = {"text": original_text, "text_keyword": text_keyword, "text_length": original_length,
                   "text_keyword_length": text_keyword_length, "text_ids": original_id,
                   "text_keyword_id": text_keyword_id, "labels": original_label, 'text_keyword_pos': text_keyword_pos}

        return Dataset


    def makeBatch(self, Dataset):
        Dataset_total = []
        text_keyword = []
        text_keyword_pos = []
        text_keyword_length = []
        text_keyword_id = []
        text = []
        original_id = []
        original_label = []
        text_length = []

        temp = {"text": text, "text_keyword": text_keyword, "text_length": text_length,
                   "text_keyword_length": text_keyword_length, "text_ids": original_id,
                   "text_keyword_id": text_keyword_id, "labels": original_label, 'text_keyword_pos': text_keyword_pos}

        for i in range(len(Dataset['text'])):
            temp["text"].append(Dataset['text'][i])
            temp["text_keyword"].append(Dataset['text_keyword'][i])
            temp["text_length"].append(Dataset['text_length'][i])
            temp["text_keyword_length"].append(Dataset['text_keyword_length'][i])
            temp["text_ids"].append(Dataset['text_ids'][i])
            temp["text_keyword_id"].append(Dataset['text_keyword_id'][i])
            temp["labels"].append(Dataset['labels'][i])
            temp["text_keyword_pos"].append(Dataset['text_keyword_pos'][i])

            if ((i+1) % self.batch_size == 0):

                store = {"text": np.array([row for row in temp['text']]),
                         "text_keyword": np.array([row for row in temp['text_keyword']]),
                         "text_length": np.array([row for row in temp['text_length']]),
                         "text_keyword_length": np.array([row for row in temp['text_keyword_length']]),
                         "text_ids": np.array([row for row in temp['text_ids']]),
                         "text_keyword_id": np.array([row for row in temp['text_keyword_id']]),
                         "labels": np.array([row for row in temp['labels']]),
                         "text_keyword_pos": np.array([row for row in temp['text_keyword_pos']])}

                Dataset_total.append(store)
                temp['text'].clear()
                temp["text_keyword"].clear()
                temp['text_length'].clear()
                temp["text_keyword_length"].clear()
                temp["text_keyword_id"].clear()
                temp['text_ids'].clear()
                temp['labels'].clear()
                temp['text_keyword_pos'].clear()

        return Dataset_total



    def shuff(self, data0, data1, shuff):
        "shuff the two dataset"

        text_keyword = []
        text_keyword_pos = []
        text_keyword_length = []
        text_keyword_id = []
        text = []
        original_id = []
        original_label = []
        text_length = []

        temp = {"text": text, "text_keyword": text_keyword, "text_length": text_length,
                "text_keyword_length": text_keyword_length, "text_ids": original_id,
                "text_keyword_id": text_keyword_id, "labels": original_label, 'text_keyword_pos': text_keyword_pos}

        temp['text'] = data0['text'] + data1['text']
        temp['text_keyword'] = data0['text_keyword'] + data1['text_keyword']
        temp['text_length'] = data0['text_length'] + data1['text_length']
        temp['text_keyword_length'] = data0['text_keyword_length'] + data1['text_keyword_length']
        temp['text_ids'] = data0['text_ids'] + data1['text_ids']
        temp['text_keyword_id'] = data0['text_keyword_id'] + data1['text_keyword_id']
        temp['labels'] = data0['labels'] + data1['labels']
        temp['text_keyword_pos'] = data0['text_keyword_pos'] + data1['text_keyword_pos']

        if shuff:
            perm = torch.randperm(len(temp['text']))
            text = [temp['text'][idx] for idx in perm]
            text_keyword = [temp['text_keyword'][idx] for idx in perm]
            text_length = [temp['text_length'][idx] for idx in perm]
            text_keyword_length = [temp['text_keyword_length'][idx] for idx in perm]
            text_keyword_id = [temp['text_keyword_id'][idx] for idx in perm]
            original_id = [temp['text_ids'][idx] for idx in perm]
            original_label = [temp['labels'][idx] for idx in perm]
            text_keyword_pos = [temp['text_keyword_pos'][idx] for idx in perm]


            data_total = {"text": text, "text_keyword": text_keyword, "text_length": text_length,
                            "text_keyword_length": text_keyword_length, "text_ids": original_id,
                            "text_keyword_id": text_keyword_id, "labels": original_label, 'text_keyword_pos': text_keyword_pos}

            return data_total

        else:
            return temp


# oracle_loader = RealDataLoader(64, 17)
# _, index2word = oracle_loader.Vocabulary('data/yelp/yelp_15/vocab')
# oracle_loader.create_batches(pos_file='data/yelp/yelp_15/sentiment.dev.sentiment.1', neg_file='data/yelp/yelp_15/sentiment.dev.0')
# # a = oracle_loader.next_batch()
# # print(a)
#
# dic = {}
# with open('/home/hsw/Datasets/pretrain_wordvec/glove.twitter.27B/glove.twitter.27B.100d.txt', 'r') as f:
#     a = f.readline()
#     print()
#     for i in f:
#         vec = i.strip().split()
#         dic[vec[0]] = vec[sentiment.1:]
#
# embd_numpy = np.zeros((9361, 100))
# for k, v in index2word.items():
#     try:
#         vec = dic[index2word[k]]
#
#     except KeyError:
#         vec = np.random.normal(scale=0.6, size=(100))
#
#     embd_numpy[k] = vec
