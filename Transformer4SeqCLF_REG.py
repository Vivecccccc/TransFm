import numpy as np
import pandas as pd
import math, os, copy
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import focal_loss
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


x_ori = np.load('d:/RE/project001/X.npy')
y_ori = np.load('d:/RE/project001/y.npy')

""" 将MOS依照规则转换为分类问题 """
y_ori[y_ori <= 3] = int(1)
y_ori[y_ori > 3] = int(0)
print('正样本占比为：{} %'.format(len(y_ori[y_ori == 1]) / len(y_ori) * 100))

x_train, x_test, y_train, y_test = train_test_split(x_ori, y_ori, test_size=0.2)

""" 将数据flatten后进行归一化 """
x_train_df = pd.DataFrame(x_train.reshape((-1, x_ori.shape[-1])))
x_test_df = pd.DataFrame(x_test.reshape((-1, x_ori.shape[-1])))

scaler_train = StandardScaler()
scaler_test = StandardScaler()

scaled_x_train_df = scaler_train.fit_transform(x_train_df)
scaled_x_test = scaler_test.fit_transform(x_test_df)

x_train = scaled_x_train_df.reshape((-1, x_ori.shape[1], x_ori.shape[-1]))
x_test = scaled_x_test.reshape((-1, x_ori.shape[1], x_ori.shape[-1]))

""" 若干类以实现Encoder """
""" 时序编码 """
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, dropout, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.dropout = layers.Dropout(dropout)
        """ 将时序编码视为不可训练的常量，初始化时序编码矩阵为零矩阵 """
        pe = tf.Variable(tf.zeros((max_len, d_model)), trainable=False)
        """ 初始化一个绝对时序矩阵，用索引表示它们的时间顺序，并展开矩阵使之多出1个维度，形状变为(max_len, 1) """
        position = tf.cast(tf.expand_dims(tf.range(0, max_len), axis=1),
                           dtype=tf.float32)
        self.position = position
"""     绝对时序矩阵初始化之后，接下来就是考虑如何将这些时序信息加入到时序编码矩阵中，
        最简单思路就是先将max_len * 1的绝对时序矩阵，变换成max_len * d_model形状，然后覆盖原来的初始时序编码矩阵即可，
        要做这种矩阵变换，就需要一个1*d_model形状的变换矩阵div_term，我们对这个变换矩阵的要求除了形状外，
        还希望它能够将自然数的绝对时序编码缩放成足够小的数字，有助于在之后的梯度下降过程中更快的收敛
        首先使用range获得一个自然数矩阵
        初始化一半即1*d_model/2 的矩阵。把它看作是初始化了两次，而每次初始化的变换矩阵会做不同的处理，第一次初始化的变换矩阵分布在正弦波上，第二次初始化的变换矩阵分布在余弦波上，
        并把这两个矩阵分别填充在时序编码矩阵的偶数和奇数位置上，组成最终的时序编码矩阵 """
        
        div_term = tf.exp(
            tf.cast(tf.range(0, d_model, 2), dtype=tf.float32) *
            -(tf.cast(math.log(10000.0) / d_model, dtype=tf.float32)))
        self.div_term = div_term
        pe[:, 0::2].assign(tf.sin(position * div_term))
        pe[:, 1::2].assign(tf.cos(position * div_term))
        """ 为了与X矩阵能够相加，拓展一个维度 """
        pe = tf.expand_dims(pe, axis=0)
        self.pe = tf.constant(pe)

    def call(self, x):
        """ 将时序矩阵与X矩阵相加 """
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)

def attention(query, key, value, dropout=None):
    d_k = query.shape[-1]
    """ 归一化点积注意力机制的实现，其中转置是将KEY部分变为(B, H, D, T)
    以与QUERY的(B, H, T, D)进行矩阵乘法，得到的矩阵是(B, H, T, T)的形式 """
    """ 其中，B为batch_size，H为自注意力头的数目，D为每一个头的维度，T为时序维度 """
    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2
                                                      ])) / math.sqrt(d_k)
    p_attn = tf.nn.softmax(scores, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    """ 返回注意力矩阵(B, H, T, T)与VALUE(B, H, T, D)的乘积，得到(B, H, T, D)的矩阵 """
    return tf.matmul(p_attn, value), p_attn

def clones(layer, N):
    """ 对层结构进行拷贝的函数实现 """
    lys = [copy.deepcopy(layer) for _ in range(N)]
    """ 检验层名并赋新的层命名 """
    assert all([lys[0].name == lys[i].name for i in range(N)])
    ori_name = lys[0].name
    neo_name_ls = [ori_name + '_' + str(i) for i in range(N)]
    for i in range(N):
        lys[i]._name = neo_name_ls[i]
    return lys

""" class MultiHeadedAttention(layers.Layer):
    def __init__(self, head, var_num, dropout=0.1, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        assert var_num % head == 0
        self.d_k = var_num // head
        self.head = head
        self.linears = clones(layers.Dense(var_num, trainable=True), 4)
        self.attn = None
        self.dropout = layers.Dropout(dropout)

    def call(self, query, key, value):
        batch_size = query.shape[0]

        query, key, value = [
            tf.transpose(tf.reshape(model(x),
                                    (batch_size, -1, self.head, self.d_k)),
                         perm=[0, 2, 1, 3])
            for model, x in zip(self.linears, (query, key, value))
        ]
        x, self_attn = attention(query, key, value, dropout=self.dropout)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]),
                       (batch_size, -1, self.head * self.d_k))

        return self.linears[-1](x) """

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, head, var_num, dropout=0.1, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        """ 检查多头数目是否能整除变量数 """
        assert var_num % head == 0
        """ 表示每个头分得的变量数目 """
        self.d_k = var_num // head
        self.head = head
        self.attn = None
        """ 实现dropout层 """
        self.dropout = layers.Dropout(rate=dropout)
        """ 将QKV分别用全连接层进行线性映射 """
        self.query_ly = layers.Dense(var_num, trainable=True, name='query')
        self.key_ly = layers.Dense(var_num, trainable=True, name='key')
        self.value_ly = layers.Dense(var_num, trainable=True, name='value')
        """ 将每个头合并后的结果利用全连接层再进行线性映射 """
        self.concat_ly = layers.Dense(var_num, trainable=True, name='concat')
        """ 层归一化的实现 """
        self.layernorm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
        
    def call(self, hidden_state):
        batch_size = hidden_state.shape[0]
        """ 将QKV在线性映射后的结果进行维度整理为(B, T, H, D)的形式，并转置为(B, H, T, D) """
        query = tf.transpose(tf.reshape(self.query_ly(hidden_state), (batch_size, -1, self.head, self.d_k)), perm=[0, 2, 1, 3])
        key = tf.transpose(tf.reshape(self.key_ly(hidden_state), (batch_size, -1, self.head, self.d_k)), perm=[0, 2, 1, 3])
        value = tf.transpose(tf.reshape(self.value_ly(hidden_state), (batch_size, -1, self.head, self.d_k)), perm=[0, 2, 1, 3])
        """ 对QKV进行注意力机制的运算得到输出x及注意力矩阵self.attn """
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        """ 将X转置为(B, T, H, D)后，合并多头（重置维度）为(B, T, H*D) """
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (batch_size, -1, self.head * self.d_k))
        """ 将多头合并后的输出利用全连接层进行映射 """
        x = self.concat_ly(x)
        """ 进行层归一化 """
        x = self.layernorm(x)
        """ 实现残差连接 """
        return hidden_state + self.dropout(x)

""" 前馈全连接层的实现，以在注意力机制后加强模型的拟合能力 """
class PositionwiseFeedForward(layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)
        """ X通过中间层w1的维度d_ff变换为d_ff维，再通过w2变换回来 """
        self.w1 = layers.Dense(d_ff, trainable=True)
        self.w2 = layers.Dense(d_model, trainable=True)
        self.dropout = layers.Dropout(rate=dropout)
        self.layernorm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
    def call(self, x):
        """ 对x进行w1线性映射后，通过激活函数ReLU与dropout层，再进行w2线性映射并dropout后进行层归一化 """
        res = self.layernorm(self.dropout(self.w2(self.dropout(tf.nn.relu(self.w1(x))))))
        """ 残差连接 """
        return x + res

""" 层归一化的实现，用于每个Encoder级别 """
class LayerNorm(layers.Layer):
    def __init__(self, var_num, eps=1e-6, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        """ 初始化a2, b2作为归一化层的缩放系数 """
        self.a2 = tf.Variable(tf.ones(var_num), trainable=True)
        self.b2 = tf.Variable(tf.zeros(var_num), trainable=True)
        """ eps的加入防止std为零导致的数值下溢 """
        self.eps = eps

    def call(self, x):
        mean = keras.backend.mean(x, axis=-1, keepdims=True)
        std = keras.backend.std(x, axis=-1, keepdims=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

""" 残差连接（子层连接）的实现 """
class SublayerConnection(layers.Layer,):
    def __init__(self, size, dropout=0.1, **kwargs):
        super(SublayerConnection, self).__init__(**kwargs)
        self.norm = LayerNorm(size, trainable=True)
        self.dropout = layers.Dropout(rate=dropout)
""" 层归一化X后将其Dropout并与X自身相加 """
    def call(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

""" 每一个Encoder的实现，以及它们堆叠的实现。后弃用。 """
""" class EncoderLayer(layers.Layer):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def call(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(layers.Layer):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Generator(layers.Layer):
    def __init__(self, d_model, cls_num):
        super(Generator, self).__init__()
        self.project = layers.Dense(cls_num)

    def call(self, x):
        return tf.nn.softmax(self.project(x), axis=-1)

class FeatureExtraction(keras.Model):
    def __init__(self, encoder, generator, position):
        super(FeatureExtraction, self).__init__()
        self.encoder = encoder
        self.generator = generator
        self.position = position

    def call(self, source):
        self.encoded = self.encode(source)
        self.avg_pooling = keras.backend.mean(self.encoded, axis=1)
        self.cls_prob = self.generator(self.avg_pooling)
        return self.cls_prob

    def encode(self, source):
        return self.encoder(source + self.position(source)) """

""" 时序维度T """
seq_len = x_ori.shape[1]
""" 特征数目D """
var_num = x_ori.shape[-1]
""" 批的大小B """
batch_size = 128
""" 多头数目H """
head = 5
""" 前馈全连接层中，中间层的维度d_ff """
d_ff = 256
""" Dropout比率 """
dropout = 0.2
""" 二分类，由于使用sigmoid激活函数输出以及binaryCrossEntropy的损失函数，分类数量设为1 """
cls_num = 1
""" Encoder的层数 """
N_enc = 6
""" 使用以下方式实现模型 """

""" 将多头自注意力层、前馈连接层复制N_enc个 """
attns = clones(MultiHeadSelfAttention(head, var_num, dropout), N_enc)
ffs = clones(PositionwiseFeedForward(var_num, d_ff, dropout), N_enc)
""" 时序编码矩阵 """
pos = PositionalEncoding(var_num, dropout)
""" 设置输入为(B, T, D)的张量 """
inputs = layers.Input(shape=(seq_len, var_num), batch_size=batch_size)
""" 将输入张量与时序编码矩阵相加 """
x = inputs + pos(inputs)
""" 将输入X经过N_enc个多头自注意力层与前馈全连接层，每次都更新其结果 """
for ly in range(N_enc):
    x = attns[ly](x)
    x = ffs[ly](x)
""" 将(B, T, D)的结果用平均池化层打平时序维度，输出为(B, D)的矩阵 """
x = layers.GlobalAveragePooling1D()(x)
""" 将这一(B, D)的矩阵利用全连接层与Dropout进行线性映射与随机丢弃 """
x = layers.Dense(d_ff)(x)
x = layers.Dropout(dropout)(x)
""" 利用全连接层与相应激活函数输出分类概率 """
outputs = layers.Dense(cls_num, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())
""" tensorflow自带的dataset类实现数据的输入管理 """
dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train.reshape((-1, 1))))
dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
dataset_val = tf.data.Dataset.from_tensor_slices((x_test, y_test.reshape((-1, 1))))
dataset_val = dataset_val.batch(batch_size, drop_remainder=True)
""" 实现多个评价指标 """
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='pr_auc', curve='PR'),
      keras.metrics.AUC(name='roc_auc', curve='ROC')
]
""" 使用Adam、binaryCrossEntropy编译模型，并进行拟合 """
model.compile(optimizer=tf.optimizers.Adam(lr=2e-4), loss='binary_crossentropy', metrics=METRICS)
history = model.fit(dataset_train, epochs=100, validation_data=dataset_val)
