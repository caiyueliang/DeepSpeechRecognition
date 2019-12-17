import keras
from keras.layers import Input, BatchNormalization, LSTM
from keras.layers import Reshape, Dense, Lambda, Dropout
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam, SGD, Adadelta
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf


def am_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        vocab_size=50,
        lr=0.0008,
        gpu_nums=1,
        is_training=True)
    return params


# =============================搭建模型====================================
class SpeechModel(object):
    """docstring for Amodel."""
    def __init__(self, args):
        self.vocab_size = args.vocab_size
        self.gpu_nums = args.gpu_nums
        self.lr = args.lr
        self.is_training = args.is_training
        print(" [vocab_size] ", self.vocab_size)
        print("   [gpu_nums] ", self.gpu_nums)
        print(" [learn_rate] ", self.lr)
        print("[is_training] ", self.is_training)

        self._model_init()

        if self.is_training:
            self._ctc_init()
            self._opt_init()

    # 模型初始化
    def _model_init(self):
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))    # 输入数据是三维: (-1, 200, 1)

        x = Reshape((-1, 200))(self.inputs)                             # 数据reshape成二维: (-1, 200)
        x = dense(512, x)                                               # 全链接层
        x = dense(512, x)                                               # 全链接层
        x = bi_gru(512, x)                                              # 双向的GRU层
        x = bi_gru(512, x)                                              # 双向的GRU层
        x = bi_gru(512, x)                                              # 双向的GRU层
        x = dense(512, x)                                               # 全链接层

        # 全链接层，输出的数据维度和字典中字的个数保持一致，激活函数是softmax
        self.outputs = dense(self.vocab_size, x, activation='softmax')

        self.model = Model(inputs=self.inputs, outputs=self.outputs)    # 模型注册
        self.model.summary()                                            # 打印网络层的信息

    # loss初始化
    def _ctc_init(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')(
            [self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs, self.input_length, self.label_length],
                               outputs=self.loss_out)

    # 优化器初始化
    def _opt_init(self):
        opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.01, epsilon=10e-8)
        if self.gpu_nums > 1:
            self.ctc_model = multi_gpu_model(self.ctc_model, gpus=self.gpu_nums)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)


# ============================模型组件=================================
# bi_gru: 是一个双向的GRU网络层，同时考虑了上下文的信息
def bi_gru(units, x, drop_rate=0.2):
    # Dropout，将输入数据按比例（0.2）随机丢弃（设置为0），防止过拟合的手段之一
    x = Dropout(drop_rate)(x)
    # units表示输出空间的维数，这里设置为512
    # return_sequences为True，表示返回完整序列，而不是只返回序列中的最后一个输出
    # kernel_initializer，表示权重矩阵的初始化方式，这里用"he_normal"
    y1 = GRU(units=units, return_sequences=True, kernel_initializer='he_normal')(x)
    # go_backwards为True，表示反向处理输入序列并返回相反的顺序。
    y2 = GRU(units=units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(x)
    # 链接y1和y2，作为双向的GRU网络层的整体输出
    y = add([y1, y2])
    return y


# dense: 全连接层，起到“分类器”的作用，将网络学到的特征映射到样本标记空间
def dense(units, x, drop_rate=0.2, activation="relu"):
    # Dropout，将输入数据按比例（0.2）随机丢弃（设置为0），防止过拟合的手段之一
    x = Dropout(drop_rate)(x)
    # units表示输出空间的维数，这里设置为512
    # activation，表示用什么激活函数，默认用"relu"
    # kernel_initializer，表示权重矩阵的初始化方式，这里用"he_normal"
    y = Dense(units, activation=activation, use_bias=True, kernel_initializer='he_normal')(x)
    return y


def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
