from datetime import datetime
import pandas as pd
import copy
import os
import random
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import torch.nn as nn
import torch.nn.functional as F

from Models.model1 import Model
from utils import EarlyStopping, load_test_data
import kmeans

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def get_filename(file_path):
    return file_path.split('/')[-1].split('.')[0]

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
scores_list = []
num_clusters = 2

def main(args, data_path, output_folder):
    batch_size = 100
    batch_total = 10000

    g, features, positive_samples, negative_samples, files, all_data = load_test_data(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Model(meta_paths=[['fa', 'af']],
                  embedding_size=71936,
                  hidden_size=args['hidden_units'],
                  out_size=4,
                  num_heads=args['num_heads'],
                  dropout=args['dropout']).to(device)

    g = g.to(device)
    features = features.to(device)

    stopper = EarlyStopping(str(os.path.basename(data_path)).split('.')[0], patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=args['weight_decay'])

    test_samples = random.sample(range(features.shape[0]), int(features.shape[0] / 3))
    _test_samples = set(test_samples)
    _positive_samples = []
    _negative_samples = []

    train_samples = list(set(range(features.shape[0])) - _test_samples)

    for i in positive_samples:
        if i[0] not in _test_samples and i[1] not in _test_samples:
            _positive_samples.append(i)

    for i in negative_samples:
        if i[0] not in _test_samples and i[1] not in _test_samples:
            _negative_samples.append(i)

    print("Original pos:", len(positive_samples), "Filtered pos:", len(_positive_samples))
    print("Original neg:", len(negative_samples), "Filtered neg:", len(_negative_samples))

    positive_samples = _positive_samples
    negative_samples = _negative_samples

    valid_samples = test_samples[:len(test_samples)//2]
    test_samples = test_samples[len(test_samples)//2:]

    batch_total = int(min(len(positive_samples), len(negative_samples), batch_total) / batch_size) * batch_size
    best_score = 0
    best_model = None
    best_epoch = 0

    for epoch in range(50):
        model.train()
        random.shuffle(positive_samples)
        random.shuffle(negative_samples)

        total_loss = 0
        for i in tqdm(range(batch_total // batch_size)):
            output = model(g, features)
            p = positive_samples[i * batch_size:(i + 1) * batch_size]
            n = negative_samples[i * batch_size:(i + 1) * batch_size]
            a, b, c, d = [], [], [], []
            for i in p:
                a.append(i[0])
                b.append(i[1])
            for i in n:
                c.append(i[0])
                d.append(i[1])
            l1 = model._forward_train_positive(output, a, b)
            l2 = model._forward_train_negative(output, c, d)
            loss = l1 + l2
            optimizer.zero_grad()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        output = model(g, features)

        linear_layer1 = nn.Linear(features.size()[1], 64).to(device)
        reshaped_tensor1 = F.relu(linear_layer1(features))
        output = torch.cat((output, reshaped_tensor1), dim=1)

        _output = torch.stack([output[i] for i in train_samples])
        cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
        score1 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

        _output = torch.stack([output[i] for i in valid_samples])
        cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
        score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

        print(f"EPOCH:{epoch}  score train:{score1:.4f} score valid:{score2:.4f}  average_loss:{total_loss / (batch_total / batch_size):.4f}")

        if score2 > best_score:
            best_score = score2
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            print(f"New best model found at epoch {epoch} with validation score: {score2:.4f}")
        if epoch - best_epoch > args['patience']:
            print(f"Early stopping at epoch {epoch}")
            break

    best_model.eval()
    output = best_model(g, features)
    linear_layer1 = nn.Linear(features.size()[1], 64).to(device)
    reshaped_tensor1 = F.relu(linear_layer1(features))
    output = torch.cat((output, reshaped_tensor1), dim=1)

    _output = output
    cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
    score0 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    _output = torch.stack([output[i] for i in valid_samples])
    cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
    score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    _output = torch.stack([output[i] for i in test_samples])
    cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
    score3 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    print(f"best_epoch:{best_epoch}   score total:{score0:.4f}  score test:{score3:.4f}   score valid:{score2:.4f}")

    final_folder = os.path.join(output_folder, timestamp)
    os.makedirs(final_folder, exist_ok=True)
    model_name = os.path.basename(data_path).split('.')[0] + ".pth"
    torch.save(best_model.state_dict(), os.path.join(final_folder, model_name))
    print(f"Saved best model to {os.path.join(final_folder, model_name)}")

    scores_list.append({
        'Train Score': score0,
        'Validation Score': score2,
        'Test Score': score3,
        'File': get_filename(data_path)
    })

if __name__ == '__main__':
    import argparse
    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-d', '--data', type=str, default='.', help='data path')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results', help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true', help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    test_result = '/home/user/wyh/DC_contribution/finally_project/code/HAN/unsupervised_output/HAN/result/mapstruct'
    output_folder = '/home/user/wyh/DC_contribution/finally_project/code/HAN/unsupervised_output/HAN/save_model/mapstruct'
    os.makedirs(test_result, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    data_folder = '/home/user/wyh/DC_contribution/finally_project/code/RFmaster/pkl_files/mapstruct/Experiment_20250221-191646'
    pkl_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]

    for pkl_file in pkl_files:
        data_path = os.path.join(data_folder, pkl_file)
        print(f"Training with data from {data_path}")
        main(args, data_path, output_folder)

    if scores_list:
        scores_df = pd.DataFrame(scores_list)
        scores_df['Average Train Score'] = scores_df['Train Score'].mean()
        scores_df['Average Validation Score'] = scores_df['Validation Score'].mean()
        scores_df['Average Test Score'] = scores_df['Test Score'].mean()
        scores_df.to_csv(os.path.join(test_result, f"scores_{timestamp}.csv"), index=False)
        print(f"Saved scores to {os.path.join(test_result, f'scores_{timestamp}.csv')}")