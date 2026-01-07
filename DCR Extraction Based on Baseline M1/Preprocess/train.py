

import json
import math
import os
import random
import re
import datetime
from collections import deque

import tensorflow as tf
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.util import tf_inspect
from transformers import BertTokenizer
from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.model_selection import KFold

# Define the basic bert class
class BertConfig(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.pop('vocab_size', 30000)
        self.type_vocab_size = kwargs.pop('type_vocab_size', 300)
        self.hidden_size = kwargs.pop('hidden_size', 768)
        self.num_hidden_layers = kwargs.pop('num_hidden_layers', 12)
        self.num_attention_heads = kwargs.pop('num_attention_heads', 12)
        self.intermediate_size = kwargs.pop('intermediate_size', 3072)
        self.hidden_activation = kwargs.pop('hidden_activation', 'gelu')
        self.hidden_dropout_rate = kwargs.pop('hidden_dropout_rate', 0.1)
        self.attention_dropout_rate = kwargs.pop('attention_dropout_rate', 0.1)
        self.max_position_embeddings = kwargs.pop('max_position_embeddings', 200)
        self.max_sequence_length = kwargs.pop('max_sequence_length', 200)


class BertEmbedding(tf.keras.layers.Layer):

    def __init__(self,**kwargs):
        super().__init__(name='BertEmbedding')
        config = BertConfig(max_sequence_length=max_len)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.token_embedding = self.add_weight('weight', shape=[self.vocab_size, self.hidden_size],
                                               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.type_vocab_size = config.type_vocab_size

        self.position_embedding = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='position_embedding'
        )
        self.token_type_embedding = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='token_type_embedding'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)

    def build(self, input_shape):
        with tf.name_scope('bert_embeddings'):
            super().build(input_shape)

    def call(self, inputs, training=False, mode='embedding'):
        # used for masked lm
        if mode == 'linear':
            return tf.matmul(inputs, self.token_embedding, transpose_b=True)

        input_ids, token_type_ids = inputs
        input_ids = tf.cast(input_ids, dtype=tf.int32)
        position_ids = tf.range(input_ids.shape[1], dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_ids.shape.as_list(), 0)

        position_embeddings = self.position_embedding(position_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        token_embeddings = tf.gather(self.token_embedding, input_ids)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).

        Returns:
            Python dictionary.
        """
        all_args = tf_inspect.getfullargspec(self.__init__).args
        config = {
            'name': self.name,
            'trainable': self.trainable,
            'vocab_size': self.vocab_size,
            'type_vocab_size': self.type_vocab_size,
            'hidden_size': self.hidden_size
        }
        if hasattr(self, '_batch_input_shape'):
            config['batch_input_shape'] = self._batch_input_shape
        config['dtype'] = policy.serialize(self._dtype_policy)
        if hasattr(self, 'dynamic'):
            # Only include `dynamic` in the `config` if it is `True`
            if self.dynamic:
                config['dynamic'] = self.dynamic
            elif 'dynamic' in all_args:
                all_args.remove('dynamic')
        expected_args = config.keys()
        # Finds all arguments in the `__init__` that are not in the config:
        extra_args = [arg for arg in all_args if arg not in expected_args]
        # Check that either the only argument in the `__init__` is  `self`,
        # or that `get_config` has been overridden:
        if len(extra_args) > 1 and hasattr(self.get_config, '_is_default'):
            raise NotImplementedError('Layer %s has arguments in `__init__` and '
                                      'therefore must override `get_config`.' %
                                      self.__class__.__name__)
        return config


# The following part is about defining relevant data path
structure_dir = '../../Dataset/train/Processed Dataset/Structure'
texture_dir = '../../Dataset/train/Processed Dataset/Texture'
picture_dir = '../../Dataset/train/Processed Dataset/Image'

# Use for texture data preprocessing
pattern = "[A-Z]"
pattern1 = '["\\[\\]\\\\]'
pattern2 = "[*.+!$#&,;{}()':=/<>%-]"
pattern3 = '[_]'

# Define basic parameters
max_len = 100

# store all data
data_set = {}

# store file name
file_name = []

# store structure information
data_structure = {}

# store texture information
data_texture = {}

# store token, position and segment information
data_token = {}
data_position = {}
data_segment = {}
# dic_content = {}

# store the content of each text
string_content = {}

# store picture information
data_picture = {}

# store content of each picture
data_image = []

# 实验部分  --  随机打乱数据
all_data = []
train_data = []
# test_data = []

structure = []
image = []
label = []
token = []
segment = []

tokenizer_path = '../Relevant Library/cased_L-12_H-768_A-12/cased_L-12_H-768_A-12'
timestamp1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
folder_name = f"Experiment_{timestamp1}"

tokenizer = BertTokenizer.from_pretrained(r"/home/user/wyh/DC_contribution/finally_project/Relevant Library/bert-base-cased")
print('Successfully load the BertTokenizer')


def preprocess_structure_data():
    for label_type in ['Readable', 'Unreadable']:
        dir_name = os.path.join(structure_dir, label_type)
        for f_name in os.listdir(dir_name):
            f = open(os.path.join(dir_name, f_name), errors='ignore')
            lines = []
            if not f_name.startswith('.'):
                file_name.append(f_name.split('.')[0])
                for line in f:
                    line = line.strip(',\n')
                    info = line.split(',')
                    info_int = []
                    count = 0
                    for item in info:
                        if count < 130:
                            info_int.append(int(item))
                            count += 1
                    info_int = np.asarray(info_int)
                    lines.append(info_int)
                f.close()
                lines = np.asarray(lines)
                if label_type == 'Readable':
                    data_set[f_name.split('.')[0]] = 0
                else:
                    data_set[f_name.split('.')[0]] = 1
                data_structure[f_name.split('.')[0]] = lines


def process_texture_data():
    for label_type in ['Readable', 'Unreadable']:
        dir_name = os.path.join(texture_dir, label_type)
        for f_name in os.listdir(dir_name):
            if f_name[-4:] == ".txt":
                list_content = []
                list_position = []
                list_segment = []
                s = ''
                segment_id = 0
                position_id = 0
                count = 0
                f = open(os.path.join(dir_name, f_name), errors='ignore')
                for content in f:
                    content = re.sub(r"([a-z]+)([A-Z]+)", r"\1 \2", content)
                    content = re.sub(pattern1, lambda x: " " + x.group(0) + " ", content)
                    content = re.sub(pattern2, lambda x: " " + x.group(0) + " ", content)
                    content = re.sub(pattern3, lambda x: " ", content)
                    list_value = content.split()
                    for item in list_value:
                        if len(item) > 1 or not item.isalpha():
                            s = s + ' ' + item
                            list_content.append(item)
                            if count < max_len:
                                list_position.append(position_id)
                                position_id += 1
                                list_segment.append(segment_id)
                            count += 1
                    segment_id += 1
                while count < max_len:
                    list_segment.append(segment_id)
                    list_position.append(count)
                    count += 1
                f.close()
                string_content[f_name.split('.')[0]] = s
                data_position[f_name.split('.')[0]] = list_position
                data_segment[f_name.split('.')[0]] = list_segment
                # dic_content[f_name.split('.')[0]] = list_content

        for sample in string_content:
            list_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(string_content[sample]))
            list_token = list_token[:max_len]
            while len(list_token) < max_len:
                list_token.append(0)
            data_token[sample] = list_token


def preprocess_picture_data():
    for label_type in ['readable', 'unreadable']:
        dir_image_name = os.path.join(picture_dir, label_type)
        for f_name in os.listdir(dir_image_name):
            if not f_name.startswith('.'):
                img_data = cv2.imread(os.path.join(dir_image_name, f_name))
                img_data = cv2.resize(img_data, (256, 256))
                result = img_data / 255.0
                data_picture[f_name.split('.')[0]] = result
                data_image.append(result)


def random_dataSet():
    count_id = 0
    while count_id < 210:
        index_id = random.randint(0, len(file_name) - 1)
        all_data.append(file_name[index_id])
        file_name.remove(file_name[index_id])
        count_id += 1
    for item in all_data:
        label.append(data_set[item])
        structure.append(data_structure[item])
        image.append(data_picture[item])
        token.append(data_token[item])
        segment.append(data_segment[item])


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_1 = true_positives / (possible_positives + K.epsilon())
    return recall_1


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_1 = true_positives / (predicted_positives + K.epsilon())
    return precision_1


def create_VST_model():
    structure_input = keras.Input(shape=(149,130), name='structure')
    structure_reshape = keras.layers.Reshape((149,130, 1))(structure_input)
    structure_conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(structure_reshape)
    structure_pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(structure_conv1)
    structure_conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(structure_pool1)
    structure_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(structure_conv2)
    structure_conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(structure_pool2)
    structure_pool3 = keras.layers.MaxPool2D(pool_size=3, strides=3)(structure_conv3)
    structure_flatten = keras.layers.Flatten()(structure_pool3)

    bert_config = BertConfig(max_sequence_length=max_len)
    token_input = keras.Input(shape=(max_len,), name='token')
    segment_input = keras.Input(shape=(max_len,), name='segment')
    texture_embedded = BertEmbedding()([token_input, segment_input])
    texture_conv1 = keras.layers.Conv1D(32, 5, activation='relu')(texture_embedded)
    texture_pool1 = keras.layers.MaxPool1D(3)(texture_conv1)
    texture_conv2 = keras.layers.Conv1D(32, 5, activation='relu')(texture_pool1)
    texture_gru = keras.layers.Bidirectional(keras.layers.LSTM(32))(texture_conv2)

    image_input = keras.Input(shape=(256, 256, 3), name='image')
    image_conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(image_input)
    image_pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image_conv1)
    image_conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(image_pool1)
    image_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image_conv2)
    image_conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(image_pool2)
    image_pool3 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image_conv3)
    image_flatten = keras.layers.Flatten()(image_pool3)

    concatenated = keras.layers.concatenate([structure_flatten, texture_gru, image_flatten], axis=-1,name='concatenated_20')


    # 原始的形式
    dense1 = keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.001),name='random')(concatenated)
    drop = keras.layers.Dropout(0.5,name='random1')(dense1)
    dense2 = keras.layers.Dense(units=16, activation='relu', name='random_detail')(drop)
    dense3 = keras.layers.Dense(1, activation='sigmoid',name='random2')(dense2)
    model = keras.Model([structure_input, token_input, segment_input, image_input], dense3)
    # rms = keras.optimizers.RMSprop(lr=0.00015)
    rms = keras.optimizers.RMSprop(lr=0.0015)
    # Models.summary()
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc', 'Recall', 'Precision', 'AUC',
                                                                      'TruePositives', 'TrueNegatives',
                                                                      'FalseNegatives', 'FalsePositives'])
    return model




def get_timestamped_filename(directory, extension='.txt'):
    """生成一个以当前时间戳命名的文件名，并包含在指定目录中"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}{extension}"
    return os.path.join(directory, filename)

def write_to_file(data, directory, filename):
    """将数据写入指定目录下的文件"""
    file_path = get_timestamped_filename(directory)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # 使用indent参数美化JSON格式

if __name__ == '__main__':
    preprocess_structure_data()
    process_texture_data()
    preprocess_picture_data()
    random_dataSet()

    # 确保数据格式正确
    label = np.asarray(label)
    structure = np.asarray(structure)
    image = np.asarray(image)
    token = np.asarray(token)
    segment = np.asarray(segment)

    print('Shape of structure data tensor:', structure.shape)
    print('Shape of image data tensor:', image.shape)
    print('Shape of token tensor:', token.shape)
    print('Shape of segment tensor:', segment.shape)
    print('Shape of label tensor:', label.shape)

    # 定义交叉验证的折数和重复次数
    k = 5
    n_repeats = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # 初始化性能指标列表和最佳模型信息
    best_model_info = {}
    best_val_acc = 0

    # 初始化性能指标列表和最佳模型信息
    best_models = deque(maxlen=5)  # 使用双端队列，限制最多保存5个模型

    # 进行交叉验证和重复训练
    for repeat in range(n_repeats):
        print(f'Repeat {repeat + 1}/{n_repeats}')
        train_accs = []
        val_accs = []
        train_losses = []
        val_losses = []

        for fold, (train_index, val_index) in enumerate(kf.split(label)):
            print(f'Fold {fold + 1}/{k}')

            # 划分训练集和验证集
            x_train_structure, x_val_structure = structure[train_index], structure[val_index]
            x_train_token, x_val_token = token[train_index], token[val_index]
            x_train_segment, x_val_segment = segment[train_index], segment[val_index]
            x_train_image, x_val_image = image[train_index], image[val_index]
            y_train, y_val = label[train_index], label[val_index]

            # 创建模型
            VST_model = create_VST_model()

            # 训练模型
            history_vst = VST_model.fit([x_train_structure, x_train_token, x_train_segment, x_train_image], y_train,
                                        validation_data=(
                                        [x_val_structure, x_val_token, x_val_segment, x_val_image], y_val),
                                        epochs=50, batch_size=16, verbose=1)

            # 记录性能指标
            train_accs.append(history_vst.history['acc'][-1])
            val_accs.append(history_vst.history['val_acc'][-1])
            train_losses.append(history_vst.history['loss'][-1])
            val_losses.append(history_vst.history['val_loss'][-1])

            # 保存当前折的模型
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            folder_path = os.path.join("../../Experimental output", folder_name)
            # 检查文件夹是否存在，如果不存在则创建
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            model_filename = f"VST_Model_Repeat_{repeat + 1}_Fold_{fold + 1}_{timestamp}.hdf5"
            model_save_path = os.path.join(folder_path, model_filename)
            VST_model.save(model_save_path)
            print(f"Model saved to {model_save_path}")

            # 更新最佳模型列表
            val_acc = history_vst.history['val_acc'][-1]
            best_models.append((model_save_path, val_acc))
            best_models = deque(sorted(best_models, key=lambda x: x[1], reverse=True), maxlen=5)  # 按验证准确率排序

        # 打印平均性能指标
        print('Average training accuracy:', np.mean(train_accs))
        print('Average validation accuracy:', np.mean(val_accs))
        print('Average training loss:', np.mean(train_losses))
        print('Average validation loss:', np.mean(val_losses))

    # 打印最佳模型信息
    print("Top 5 best Models:")
    for idx, (path, val_acc) in enumerate(best_models):
        print(f"Model {idx + 1}: {path} with validation accuracy: {val_acc}")

    # 删除效果不好的模型
    folder_path = os.path.join("../../Experimental output", folder_name)
    all_model_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                       f.startswith("VST_Model")]
    for model_path in all_model_paths:
        if model_path not in [path for path, _ in best_models]:
            os.remove(model_path)
            print(f'Deleted model {os.path.basename(model_path)}')

    # 重命名最佳模型
    for idx, (path, _) in enumerate(best_models):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_filename = f"Top{idx + 1}.hdf5"
        best_model_final_path = os.path.join(folder_path, model_filename)
        os.rename(path, best_model_final_path)
        print(f'Best model {idx + 1} saved to {best_model_final_path}')