## 词袋模型
- 运行环境 `python 3.7`
- 安装包：`nltk 3.3.0`或以上 `sklearn 0.19.22`或以上
- 关联文件： `Weibo`文件夹和`Weibo.txt`文件，可通过百度网盘（链接：https://pan.baidu.com/s/1H5Ujwhy_pU8B2PW7E8O_UA 提取码：5qjh）下载，存放在data文件夹下。
- 程序运行：在`1_bag_of_word.ipynb`中运行查看结果。

## 特征抽取
- 运行环境 `python 3.7`
- 安装包：`nltk 3.4` `sklearn 0.20.0` `snownlp 0.12.3` `jieba 0.39` `gensim 3.6.0`
- 特征抽取：将原数据放在`origin_data`文件夹下，创建`DATA`文件夹来存储结果。
- 运行`python Feature_label.py`来抽取特征，抽取的结果将存储在`DATA`文件夹下。

## 特征工程
- 运行环境 `python 3.7`
- 安装包：`numpy 1.15.4` `sklearn 0.20.0`
- 关联文件： `feature_data`文件夹下的`Label.npy`,`Feature.npy`文件。
- 程序运行：在`2_feature_engineering.ipynb`中运行查看结果。

## 时序模型
- 运行环境 `python 3.7`
- 安装包：`numpy 1.15.4` `sklearn 0.20.0`
- 关联文件： `Label.npy`,`index_train.npy`和`index_test.npy`文件，以及通过特征抽取得到的`DATA`文件夹，放在同一个目录下。
可通过百度网盘（链接：https://pan.baidu.com/s/1fxCKY2m_7CIs_zcQCB55Jg 提取码：st5a）下载
- 程序运行：运行`python Model_TS_Final.py`

## 循环神经网络
- 运行环境 `python 3.7`
- 安装包：`pytorch 0.4.1`
- 关联文件：`rnn_data`文件夹中的`rumor_nonrumor_vocabsize_10000_N_20_topk_10.npz`文件为预处理后的输入，`vocab_embed_10000.npy`为根据腾讯AI Lab推出的词向量提取的词汇表中的词语的词向量。
- 程序运行：在`3_RNN.ipynb`中运行查看结果。
