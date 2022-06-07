import os


def word2sent(word_list):
    sent_list = []
    tmp_sent = []
    for i in range(len(word_list)):
        if word_list[i]["ID"] == '1':
            if len(tmp_sent) > 0:
                sent_list.append(tmp_sent)
                tmp_sent = []
        tmp_sent.append(word_list[i])
    return sent_list


def clean_data(line, features):
    tmp_list = line[:-1].split("\t")
    tmp_data = {}
    # corrupted data
    if len(tmp_list) < 15:
        return tmp_data
    # complete data
    ## add features
    for i in range(len(features)-1):
        tmp_data[features[i]] = tmp_list[i]
    ## add labels
    tmp_labels = []
    for i in range(len(features)-1,len(tmp_list)):
        if tmp_list[i] != "_":
            tmp_labels.append(tmp_list[i])
    if len(tmp_labels):
        tmp_data[features[-1]] = tmp_labels
    else:
        tmp_data[features[-1]] = ["_"]
    return tmp_data


def load_data(path):
    features = ["ID", "FORM", "LEMMA", "PLEMMA", "POS", "PPOS", "FEAT", "PFEAT", \
                "HEAD", "PHEAD", "DEPREL", "PDEPREL", "FILLPRED", "PRED", "APREDs"]
    # load two txt files
    train_data, test_data = [], []
    with open(os.path.join(path, "CoNLL2009-ST-English-train.txt"), "r") as f:
        for line in f:
            tmp_data = clean_data(line, features)
            if len(tmp_data) > 1: train_data.append(tmp_data)
            # break
    with open(os.path.join(path, "CoNLL2009-ST-English-development.txt"), "r") as f:
        for line in f:
            tmp_data = clean_data(line, features)
            if len(tmp_data) > 1: test_data.append(tmp_data)
            # break
    return word2sent(train_data), word2sent(test_data)


def print_statistics(data_list):
    features = ["ID", "FORM", "LEMMA", "PLEMMA", "POS", "PPOS", "FEAT", "PFEAT", \
                "HEAD", "PHEAD", "DEPREL", "PDEPREL", "FILLPRED", "PRED", "APREDs"]
    for f in features:
        tmp_dict = {}
        if f != "APREDs":
            for s in data_list:
                for w in s:
                    if w[f] not in tmp_dict:
                        tmp_dict[w[f]] = 1
                    else:
                        tmp_dict[w[f]] += 1
        else:
            for s in data_list:
                for w in s:
                    tmp_labels = w[f]
                    for l in tmp_labels:
                        if l not in tmp_dict:
                            tmp_dict[l] = 1
                        else:
                            tmp_dict[l] += 1
        print(f, tmp_dict)
    return


def preprocessing(data_list, features):
    feats, labels = [], []
    # sentence-level
    for s in data_list:
        tmp_s_f, tmp_s_l = [], []
        # word-level
        for w in s:
            # no label => not argument => ignore
            if w["APREDs"] == "_": continue
            tmp_s_f.append({k:v for k, v in w.items() if k in features})
            tmp_s_l.append(w["APREDs"][0])
        feats.append(tmp_s_f)
        labels.append(tmp_s_l)
    return feats, labels