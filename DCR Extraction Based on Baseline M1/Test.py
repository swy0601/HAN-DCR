import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from tensorflow import timestamp
from tensorflow.keras.models import load_model
from transformers import BertTokenizer
from Models.model1 import Model
from utils import load_test_data
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow import keras
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
    def __init__(self, **kwargs):
        super().__init__(name='BertEmbedding')
        config = BertConfig(max_sequence_length=100)
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

    def call(self, inputs, training=False, mode='embedding'):
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

def evaluate_model(model, test_structure, test_token, test_segment, test_image, test_label):
    """
    评估模型并返回性能指标。
    """
    y_pred = model.predict([test_structure, test_token, test_segment, test_image], verbose=0)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()

    test_accuracy = accuracy_score(test_label, y_pred_class)
    test_precision = precision_score(test_label, y_pred_class)
    test_recall = recall_score(test_label, y_pred_class)
    test_f1 = f1_score(test_label, y_pred_class)
    test_auc = roc_auc_score(test_label, y_pred)
    test_mcc = matthews_corrcoef(test_label, y_pred_class)

    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'auc': test_auc,
        'mcc': test_mcc
    }

def print_metrics_table(metrics_list):
    """
    打印评估指标表格。
    """
    print("\nModel Evaluation Metrics:")
    print(f"{'Model Name':<40} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'AUC':<10} | {'MCC':<10}")
    print("-" * 120)
    for metrics in metrics_list:
        print(f"{metrics['model_name']:<40} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['auc']:.4f} | {metrics['mcc']:.4f}")

def write_metrics_to_file(metrics_list, average_metrics, output_file):
    """
    将评估指标（包括单个模型的评估结果和平均指标）写入文件。
    """
    with open(output_file, 'w') as f:
        # 写入表头
        f.write("Model Evaluation Metrics:\n")
        f.write(f"{'Model Name':<40} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'AUC':<10} | {'MCC':<10}\n")
        f.write("-" * 120 + "\n")

        # 写入每个模型的评估结果
        for metrics in metrics_list:
            f.write(f"{metrics['model_name']:<40} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['auc']:.4f} | {metrics['mcc']:.4f}\n")

        # 写入平均指标
        f.write("\nAverage Model Evaluation Metrics:\n")
        f.write(f"{'Average':<40} | {average_metrics['accuracy']:.4f} | {average_metrics['precision']:.4f} | {average_metrics['recall']:.4f} | {average_metrics['f1_score']:.4f} | {average_metrics['auc']:.4f} | {average_metrics['mcc']:.4f}\n")

def main(data_folder, model_folder, han_folder, output_folder,folder_name):
    # Load all .pkl files from the data folder
    data_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
    metrics_list = []
    average_metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'auc': 0,
        'mcc': 0
    }

    for data_file in data_files:
        data_path = os.path.join(data_folder, data_file)
        print(f"Processing data file: {data_path}")

        # Load data
        g, features, positive_sample, negative_sample, files, label = load_test_data(data_path)

        features_tensor_numpy = features.cpu().detach().numpy()

        # Load corresponding HAN Models
        han_model_path = os.path.join(han_folder, data_file.replace('.pkl', '.pth'))
        han_model = Model(meta_paths=[['fa', 'af']],
                          embedding_size=71936,
                          hidden_size=8,
                          out_size=4,
                          num_heads=[8],
                          dropout=0.6)
        han_model.load_state_dict(torch.load(han_model_path))
        han_model.eval()

        # Get HAN features
        with torch.no_grad():
            han_output = han_model(g, features).cpu().numpy()

        # Load corresponding VST Models
        model_file = data_file.replace('.pkl', '.hdf5')
        model_path = os.path.join(model_folder, model_file)
        vst_model = load_model(model_path, custom_objects={'BertEmbedding': BertEmbedding})




        linear_layer1 = nn.Linear(features.size()[1], 64)
        reshaped_tensor1 = F.relu(linear_layer1(features))

        mid_tensor = torch.cat((torch.tensor(han_output), reshaped_tensor1), dim=1)
        linear_layer2 = nn.Linear(mid_tensor.size()[1], 64)
        reshaped_tensor = F.relu(linear_layer1(features))
        reshaped_tensor_numpy = reshaped_tensor.cpu().detach().numpy()


        # Build new Models
        new_input = tf.keras.Input(shape=reshaped_tensor_numpy.shape[1:], name='new_input')


        dense2 = vst_model.get_layer('random_detail')
        dense3 = vst_model.get_layer('random2')
        x = dense2(new_input)
        x = dense3(x)
        new_model = tf.keras.Model(inputs=new_input, outputs=x)

        # Evaluate Models
        y_pred = new_model.predict(reshaped_tensor_numpy)
        y_pred_class = (y_pred > 0.5).astype(int).flatten()

        test_accuracy = accuracy_score(label, y_pred_class)
        test_precision = precision_score(label, y_pred_class)
        test_recall = recall_score(label, y_pred_class)
        test_f1 = f1_score(label, y_pred_class)
        test_auc = roc_auc_score(label, y_pred)
        test_mcc = matthews_corrcoef(label, y_pred_class)

        metrics = {
            'model_name': data_file,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'auc': test_auc,
            'mcc': test_mcc
        }
        metrics_list.append(metrics)

        # 累加每个指标
        for key in average_metrics:
            average_metrics[key] += metrics[key]

    # 计算平均指标
    num_models = len(data_files)
    for key in average_metrics:
        average_metrics[key] /= num_models

    # 打印评估结果表格
    print_metrics_table(metrics_list)
    print("\nAverage Model Evaluation Metrics:")
    print(f"{'Average':<40} | {average_metrics['accuracy']:.4f} | {average_metrics['precision']:.4f} | {average_metrics['recall']:.4f} | {average_metrics['f1_score']:.4f} | {average_metrics['auc']:.4f} | {average_metrics['mcc']:.4f}")

    # 将评估结果写入文件
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file1 = folder_name + '_' + timestamp +'.txt'
    output_file = os.path.join("/home/user/wyh/DC_contribution/finally_project/code/HAN/test_result", file1)
    write_metrics_to_file(metrics_list, average_metrics, output_file)
    print(f"Evaluation metrics written to {output_file}")

if __name__ == '__main__':
    folder_name = 'Experiment_20250221-191646'
    data_folder = "/home/user/wyh/DC_contribution/finally_project/code/RFmaster/pkl_files/mlkit_label/" + folder_name
    model_folder = "/home/user/wyh/DC_contribution/finally_project/Experimental output/" + folder_name
    # han_folder = "/home/user/wyh/DC_contribution/finally_project/code/HAN/Experimental output/" + folder_name
    # han_folder = "/home/user/wyh/DC_contribution/finally_project/code/HAN/Experimental output/" + folder_name
    han_folder = "/home/user/wyh/DC_contribution/finally_project/code/HAN/unsupervised_output/HAN/save_model/mlkit/20250929-193056"


    output_folder = "/home/user/wyh/DC_contribution/finally_project/code/HAN/test_result"
    os.makedirs(output_folder, exist_ok=True)
    main(data_folder, model_folder, han_folder, output_folder,folder_name)