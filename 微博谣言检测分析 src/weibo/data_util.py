import os
import numpy as np
import time
from tqdm import tqdm

from preprocess import Cut

class DATA:
    
    def __init__(self, vocab_size, N, topk):
        self.word_array_file = "/home/lchen/Datasets/embedding/Tencent_AI_Chinese/Tencent_AILab_ChineseWord.npy"
        self.embed_array_file = "/home/lchen/Datasets/embedding/Tencent_AI_Chinese/Tencent_AILab_ChineseEmbedding.npy"
        self.preprocess_path = "./rnn_data"
        self.vocab_size = vocab_size
        self.N = N
        self.topk = topk
        
    def load_data(self):
        data_file = "rumor_nonrumor_vocabsize_%d_N_%d_topk_%d.npz" % (self.vocab_size, self.N, self.topk)
        if data_file in os.listdir(self.preprocess_path):
            print("---------- load preprocessed data ----------")
            

        else:
            print("---------- construct preprocessed data ----------")
            cut = Cut(vocabulary_size=self.vocab_size, N=self.N, topk=self.topk, pre_path=self.preprocess_path)
            cut.divide_all()
        data_file = os.path.join(self.preprocess_path,
                "rumor_nonrumor_vocabsize_%d_N_%d_topk_%d.npz" % (self.vocab_size, self.N, self.topk))
        npz = np.load(data_file)
        vocab, nonrumor, nonrumor_len, rumor, rumor_len = npz["arr_0"], npz["arr_1"], npz["arr_2"], npz["arr_3"], npz["arr_4"]
        self.vocab = vocab
        x = np.vstack((nonrumor, rumor))
        x_len = np.concatenate((nonrumor_len, rumor_len))
        y = np.concatenate((np.zeros(nonrumor.shape[0]), np.ones(rumor.shape[0])))
        print("x:", x.shape, "x_len:", x_len.shape, "y:", y.shape)
        return x, x_len, y

    def load_embed(self):
        embed_file = "vocab_embed_%d.npy" % self.vocab_size
        if embed_file in os.listdir(self.preprocess_path):
            print("---------- load saved pretrained embedding weight ----------")
            vocab_embed = np.load(os.path.join(self.preprocess_path, embed_file))
        else:
            print("---------- construct pretrained embedding weight ----------")
            t1 = time.time()
            word_array_read = np.load(self.word_array_file)
            embed_array_read = np.load(self.embed_array_file)
            print("load Tencent pretrained embedding consume: %.2f s" % (time.time()-t1))
            word_indices = {item[1]:item[0] for item in enumerate(word_array_read)}
            vocab_indices = {item[1]:item[0] for item in enumerate(self.vocab, 1)}
            vocab_embed = np.zeros((self.vocab.size+2, 200)) # vocab.shape+2 0 & unknown words
            i = 0
            for indice in tqdm(vocab_indices.values(), ncols=50):
                try:
                    vocab_embed[indice] = embed_array_read[word_indices[vocab[indice]]]
                except:
                    vocab_embed[indice] = np.random.rand(1, 200)
            np.save(os.path.join(self.preprocess_path, "vocab_embed_%d.npy" % self.vocab.size), vocab_embed)
        print("vocab_embed:", vocab_embed.shape)
        return vocab_embed 

if __name__ == "__main__":
    vocab_size = 10000
    N = 20
    topk = 10
    data = Data(vocab_size=vocab_size, N=N, topk=topk)
    x, x_len, y = data.load_data()
    pretrained_weight = data.load_embed()