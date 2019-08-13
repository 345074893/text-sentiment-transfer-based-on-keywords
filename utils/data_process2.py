import torch


def Vocabulary(voc_file):

    word2index = {}
    n = 0
    with open(voc_file, 'r') as voc:
        for i in voc:
            word2index[i.strip('\n')] = n
            n += 1

    index2word = dict(zip(word2index.values(), word2index.keys()))
    return word2index, index2word

def makeData(srcFile, word_to_id, label, max_length = 17, if_shuff=True):

    original_text = []
    original_id = []
    original_label = []
    original_length = []


    print('Processing %s  ...' % srcFile)
    srcF = open(srcFile, "r")
    srcSet = srcF.readlines()


    for i in range(len(srcSet)):
        if i % 5000 == 0:
            print("now: ", i)
        id_line = []
        srcSet[i] = srcSet[i].strip()

        original_label.append(label)
        text_line = ["<START>"] + srcSet[i].split() + ["<END>"]
        original_text.append(text_line)
        original_length.append(len(text_line))

        for j in range(len(original_text[i])):
            try:
                id = word_to_id[original_text[i][j]]

                # if original_mask_text[i][j] == '<mask>':
                #     mask_id = 0
                # else:
                #     mask_id = sentiment.1
            except KeyError:
                id = 3
            id_line.append(id)
        original_id.append(id_line)

    print('... padding')
    for i in range(len(original_text)):
        if original_length[i] < max_length:
            for j in range(max_length - original_length[i]):
                original_text[i].append("<PAD>")
                original_id[i].append(0)

    Dataset = {"text": original_text, "length": original_length,
               "text_ids": original_id, "labels": original_label}

    return Dataset


def makeBatch(Dataset, batch_size):
    Dataset_total = []
    text = []
    length = []
    text_ids = []
    labels = []

    temp = {"text": text, "length": length, "text_ids": text_ids, "labels": labels}

    for i in range(len(Dataset['text'])):
        temp["text"].append(Dataset['text'][i])
        temp["length"].append(Dataset['length'][i])
        temp["text_ids"].append(Dataset['text_ids'][i])
        temp["labels"].append(Dataset['labels'][i])

        if ((i+1) % batch_size == 0):

            store = {"text": [row for row in temp['text']],"length": [row for row in temp['length']],
                     "text_ids": [row for row in temp['text_ids']], "labels": [row for row in temp['labels']]}
            Dataset_total.append(store)
            temp['text'].clear()
            temp['length'].clear()
            temp['text_ids'].clear()
            temp['labels'].clear()

    return Dataset_total



def shuff(data0, data1):
    "shuff the two dataset"

    Dataset_total = {}
    text = []
    length = []
    text_ids = []
    labels = []

    temp = {"text": text, "length": length, "text_ids": text_ids, "labels": labels}
    temp['text'] = data0['text'] + data1['text']
    temp['length'] = data0['length'] + data1['length']
    temp['text_ids'] = data0['text_ids'] + data1['text_ids']
    temp['labels'] = data0['labels'] + data1['labels']


    perm = torch.randperm(len(temp['text']))
    original_id = [temp['text_ids'][idx] for idx in perm]
    original_label = [temp['labels'][idx] for idx in perm]
    original_length = [temp['length'][idx] for idx in perm]
    original_text = [temp['text'][idx] for idx in perm]


    data_total = {"text": original_text, "labels": original_label,
                  "length":original_length, "text_ids":original_id}

    return data_total

# word2index, index2word = Vocabulary('data/yelp/yelp_15/vocab')
# Dataset1 = makeData('data/yelp/yelp_15/sentiment.dev.sentiment.1', word2index, label=sentiment.1)
# Dataset0 = makeData('data/yelp/yelp_15/sentiment.dev.0', word2index, label=0)
# data_total = shuff(Dataset1, Dataset0)
# data_total = makeBatch(data_total, 5)
