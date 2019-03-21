import os
import json
from langconv import *
import jieba
import numpy as np
import re
import nltk

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

# extract text and word

class Simple:

    def __init__(self):
        self.original_path = "/home/lchen/Datasets/weibo/Weibo"
        self.record_path = "/home/lchen/Datasets/weibo/Weibo_text"
        self.record_word_path = "/home/lchen/Datasets/weibo/Weibo_word"

    # convert traditional characters to simple
    def tradition2simple(self, line):
        line = Converter("zh-hans").convert(line)
        line = line.encode("utf-8")
        return line.decode("utf-8")

    def cut_simple_sentence(self, line):
        simple_sentence = self.tradition2simple(line)
        return " ".join(jieba.cut(simple_sentence)).split()

    # extract time and text of every event
    def extract_t_text(self, original_path, record_path, record_word_path):
        original_path = self.original_path
        record_path = self.record_path
        record_word_path =self.record_word_path
        filenames = os.listdir(original_path)
        original_filenames = [os.path.join(original_path, filename) for filename in filenames]
        record_filenames = [os.path.join(record_path, filename) for filename in filenames]
        record_word_filenames = [os.path.join(record_word_path, filename) for filename in filenames]
        for i in tqdm(range(len(filenames))):
            with open(original_filenames[i], "r", encoding="utf-8") as f:
                event = json.load(f)
            record_event = [{"t": item["t"], "text": self.tradition2simple(item["text"])} for item in event]
            record_word = [{"t": item["t"], "word": self.cut_simple_sentence(item["text"])} for item in event]
            # print(record_event)
            # print(record_word)
            with open(record_filenames[i], "w", encoding="utf-8") as f:
                json.dump(record_event, f, ensure_ascii=False)
            with open(record_word_filenames[i], "w", encoding="utf-8") as f:
                json.dump(record_word, f, ensure_ascii=False)

class TFIDF:
    def __init__(self, pre_path="./preprocess"):
        self.pre_path = pre_path
        
    # 将每个文档中的word整合起来
    def load_text(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        word_list = []
        for item in data:
            word_list.extend(item["word"])
        return word_list

    # 对每个文档中的词语的tf-idf做整合
    def integrate_tfidf(self, word_bag, weight, word_list):
        word_freq_dict = nltk.FreqDist(word_list)
        word_tfidf_freq = []
        dtype = [("Word", "U20"), ("tfidf", float), ("freq", int)]
        for i in tqdm(range(weight.shape[1]), ncols=50):  # weight.shape (文档数，词数)
            tfidf_array = weight[:, i]
            word_tfidf_freq.append((word_bag[i], np.mean(tfidf_array), word_freq_dict[word_bag[i]]))
        word_tfidf_freq = np.array(word_tfidf_freq, dtype=dtype)
        word_tfidf_freq = np.sort(word_tfidf_freq, order=["tfidf", "freq"])[::-1]
        return word_tfidf_freq

    # 计算每个词语的tf-idf平均值 词频计算出来 使用sklearn计算tf-idf会将一个字的词语过滤掉
    def word_bag(self, path):
        word_list = []
        text_list = []
        filenames = [os.path.join(path, filename) for filename in os.listdir(path) if re.match(r"\d+\.json", filename)]
        for filename in tqdm(filenames, ncols=50):
            word_tempt = self.load_text(filename)
            word_list.extend(word_tempt)
            text_list.append(" ".join(word_tempt))
        print("---------- counting tfidf ----------")
        vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(text_list))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        word_bag = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        print("---------- integrate tfidf and frequency ----------")
        word_tfidf_freq = self.integrate_tfidf(word_bag, weight, word_list)
        np.save(os.path.join(self.pre_path, "word_tfidf_freq.npy"), word_tfidf_freq)


class Cut:

    def __init__(self, vocabulary_size=10000, N=20, topk=10, pre_path="./preprocess"):
        self.vocabulary_size = vocabulary_size
        self.N = N  # base num for cutting the timeline
        self.max_seq_len = self.N * 2
        self.topk = topk  # select topk words in a time batch according to tfidf
        self.weibo_label_path = "/home/lchen/Datasets/weibo/Weibo.txt"
        self.weibo_word_path = "/home/lchen/Datasets/weibo/Weibo_word"
        self.pre_path = pre_path
        self.word_tfidf_freq_file = "word_tfidf_freq.npy"

    def load_vocab(self):
        print("---------- load vocabulary ----------")
        if self.word_tfidf_freq_file not in os.listdir(self.pre_path):
            tfidf = TFIDF(pre_path=self.pre_path)
            tfidf.word_bag(self.weibo_word_path)
        word_tfidf_freq = np.load(os.path.join(self.pre_path, self.word_tfidf_freq_file))            
        word_tfidf_freq = word_tfidf_freq[:self.vocabulary_size]
        self.vocab = [item[0] for item in word_tfidf_freq]
        self.vocab_indices = {item[1]: item[0] for item in enumerate(self.vocab, 1)}

    def load_label(self):  # nonrumor-0 rumor-1
        print("---------- load labels ----------")
        with open(self.weibo_label_path, "r") as f:
            s = f.readlines()
        id_list = [[], []]
        for line in s:
            ID_label = [item.split(':')[1] for item in line.split('\t')[:2]]
            if ID_label[1] == '0':
                id_list[0].append(ID_label[0])
            if ID_label[1] == '1':
                id_list[1].append(ID_label[0])
        self.id_list = id_list

    def divide_time(self, timeList, N):
        t = len(timeList)
        if t < N:
            return [[item] for item in timeList]
        else:
            batchTime = []
            history = [1]
            interval = (timeList[-1] - timeList[0]) / N
            while len(batchTime) < N and history[-1] < t:
                batchTime = []
                j = 0
                batchTime.append([])
                batchTime[j] = []
                for time in timeList:
                    batchTime[j].append(time)
                    if batchTime[j][-1] - batchTime[j][0] >= interval:
                        j += 1
                        batchTime.append([])
                batchTime = [item for item in batchTime if item]
                t = len(batchTime)
                interval = interval * 0.5
            return batchTime

    def divide_weibo(self, ID, topk):
        filename = os.path.join(self.weibo_word_path, "%s.json" % ID)
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        time_list, word_list, word_bag = [], [], []
        for item in data:
            words = [self.vocab_indices.get(word, self.vocabulary_size + 1)
                     for word in item["word"]]  # 在vocab中返回indice 在字典外返回vocabulary_size+1
            time_list.append(item["t"])
            word_list.append(words)
        time_batch = self.divide_time(time_list, N=self.N)
        i = 0
        word_batch = []
        for batch in time_batch:
            word_batch_tempt = []
            for item in batch:
                word_batch_tempt.extend(word_list[i])
                i += 1
            word_batch_tempt = np.array(word_batch_tempt)
            # word_batch_tempt = np.unique(word_batch_tempt)         # 获取词语indice集合
            word_batch_tempt = np.sort(word_batch_tempt)  # 从小到大排序
            if word_batch_tempt.size >= topk:
                word_batch_tempt = word_batch_tempt[:topk]  # 取前topk个
            else:
                word_batch_tempt = np.pad(word_batch_tempt,
                                          (0, topk - word_batch_tempt.size), 'constant',
                                          constant_values=0)  # 不够topk 补0
            word_batch.append(word_batch_tempt)
        return np.array(word_batch)

    def divide_all(self):
        self.load_vocab()
        self.load_label()
        print("---------- divide nonrumor weibo ----------")
        nonrumor, nonrumor_len = [], []
        for ID in tqdm(self.id_list[0], ncols=50):
            word_batch = self.divide_weibo(ID, topk=self.topk)
            nonrumor_len.append(word_batch.shape[0])
            word_batch = np.pad(word_batch, ((0, self.max_seq_len - word_batch.shape[0]), (0, 0)), 'constant',
                                constant_values=0)
            nonrumor.append(word_batch)
        print("---------- divide rumor weibo ----------")
        rumor, rumor_len = [], []
        for ID in tqdm(self.id_list[1], ncols=50):
            word_batch = self.divide_weibo(ID, topk=self.topk)
            rumor_len.append(word_batch.shape[0])
            word_batch = np.pad(word_batch, ((0, self.max_seq_len - word_batch.shape[0]), (0, 0)), 'constant',
                                constant_values=0)
            rumor.append(word_batch)
        record_file = "./preprocess/rumor_nonrumor_vocabsize_%d_N_%d_topk_%d.npz" % (
        self.vocabulary_size, self.N, self.topk)
        np.savez(record_file, self.vocab,
                 np.asarray(nonrumor), np.asarray(nonrumor_len),
                 np.asarray(rumor), np.asarray(rumor_len))  # save data
        print("data saved at %s" % record_file)

if __name__ == "__main__":
    cut = Cut()
    cut.divide_all()
