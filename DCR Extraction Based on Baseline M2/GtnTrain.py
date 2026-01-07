
import copy
import os.path
import pandas as pd
import random
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import kmeans
from sklearn.metrics import silhouette_score
from model.gtn.model import GTN
from model.model1 import Model
from utils import EarlyStopping, load_test_data
from datetime import datetime

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

 # 提取文件名的函数
def get_filename(file_path):
    return file_path.split('/')[-1].split('.')[0]

num_clusters = 2
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
scores_list = []

def main(args, data_path, output_folder):
    g, features, positive_samples, negative_samples ,files,_= load_test_data(data_path)
    batch_size = 100
    batch_total = 10000

    b=g.adj(etype='af').coalesce().indices()
    num_nodes=g.num_nodes()
    af=torch.zeros(size=(num_nodes,num_nodes))
    fa=torch.zeros(size=(num_nodes,num_nodes))

    for i in range(b[0].__len__()):
        af[b[0][i].item()+g.num_nodes(ntype="file")][b[1][i].item()]=1
        fa[b[1][i].item()][b[0][i].item()+g.num_nodes(ntype="file")]=1
        # edge[b[1][i].item()][b[0][i].item() + g.num_nodes(ntype="file")] = 1

    edges=[fa,af]

    for i,af in enumerate(edges):
        if i ==0:
            A = af.type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,af.type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device=torch.device("cpu")
    node_features=torch.cat([features,torch.zeros(size=(g.num_nodes("author"),features.shape[1]))],dim=0)
    features=node_features.to(device)


    # features=torch.randint(0,1,size=features.shape).type(torch.FloatTensor)


    g=A.to(device)

    gtn = GTN(num_edge=A.shape[-1],
              num_channels=2,
              w_in=node_features.shape[1],
              w_out=128,
              # num_class=num_classes,
              num_layers=2,
              norm=False)
    model = Model(meta_paths=[['fa', 'af']],
                  embedding_size=71936,
                  hidden_size=args['hidden_units'],
                  out_size=4,
                  num_heads=args['num_heads'],
                  dropout=args['dropout'],
                  backbone=gtn).to(device)


    best_score=0
    best_model=None

    stopper = EarlyStopping(str(os.path.basename(data_path)).split('.')[0], patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                 weight_decay=args['weight_decay'])

    import random

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

    print(positive_samples.__len__(), _positive_samples.__len__())
    print(negative_samples.__len__(), _negative_samples.__len__())

    positive_samples = _positive_samples
    negative_samples = _negative_samples

    valid_samples = test_samples[0:int(test_samples.__len__() / 2)]
    test_samples = test_samples[int(test_samples.__len__() / 2):]

    batch_total = int(
        min(positive_samples.__len__(), negative_samples.__len__(), batch_total) / batch_size) * batch_size
    best_epoch = 0

    for epoch in range(50):
        model.train()
        random.shuffle(positive_samples)
        random.shuffle(negative_samples)

        total_loss = 0
        for i in tqdm(range(int(batch_total / batch_size))):
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
            # loss.requires_grad_(True)
            optimizer.zero_grad()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        # val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(Models, g, features, labels, val_mask, loss_fcn)
        # early_stop = stopper.step(val_loss.data.item(), val_acc, Models)
        model.eval()
        output = model(g, features)

        _output=torch.tensor([output[i].cpu().detach().numpy() for i in train_samples]).to(device)
        cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
        score1 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())


        _output=torch.tensor([output[i].cpu().detach().numpy() for i in valid_samples]).to(device)
        cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
        score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

        print("EPOCH:{}  score train:{} score valid:{}  average_loss:".format(epoch, score1,score2), total_loss / (batch_total / batch_size))

        # print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
        #       'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
        #     epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))
        if score2 > best_score:
            best_score = score2
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            print(f"New best model found at epoch {epoch} with validation score: {score2}")
        if epoch - best_epoch > args['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
        # stopper.save_checkpoint(Models)
    best_model.eval()
    output = best_model(g, features)

    _output = output
    cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
    score0 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())


    _output = torch.tensor([output[i].cpu().detach().numpy() for i in valid_samples]).to(device)
    cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
    score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    _output = torch.tensor([output[i].cpu().detach().numpy() for i in test_samples]).to(device)
    cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
    score3 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    print("best_epoch:{}   score total:{}  score test:{}   score valid:{} ".format(best_epoch, score0, score3, score2))

    final_folder = os.path.join(output_folder, timestamp)
    os.makedirs(final_folder, exist_ok=True)
    model_name = os.path.basename(data_path).split('.')[0] + ".pth"
    torch.save(best_model.state_dict(), os.path.join(final_folder, model_name))
    print(f"Saved best model to {os.path.join(output_folder, model_name)}")

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
    parser.add_argument('-d','--data',type=str,default='.',help='data path')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)
    # DATA_PATH=args.__getitem__('data')
    # output_folder = os.path.join("Experimental output", folder_name)

    test_result = '/home/user/wyh/DC_contribution/finally_project/code/HAN/unsupervised_output/GTN/result/mlkit'
    output_folder ='/home/user/wyh/DC_contribution/finally_project/code/HAN/unsupervised_output/GTN/save_model/mlkit'
    os.makedirs(test_result, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Load all .pkl files in the folder and train Models
    data_folder = '/home/user/wyh/DC_contribution/finally_project/code/RFmaster/pkl_files/mlkit_unlabel/Experiment_20250225-111739'
    pkl_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]

    for pkl_file in pkl_files:
        data_path = os.path.join(data_folder, pkl_file)
        print(f"Training with data from {data_path}")
        main(args, data_path, output_folder)

    if scores_list:
        scores_df = pd.DataFrame(scores_list)

        # 计算每列的平均值
        train_scores = scores_df['Train Score'].mean()
        validation_scores = scores_df['Validation Score'].mean()
        test_scores = scores_df['Test Score'].mean()

        # 将平均值添加到DataFrame中
        scores_df['Average Train Score'] = train_scores
        scores_df['Average Validation Score'] = validation_scores
        scores_df['Average Test Score'] = test_scores

        scores_df.to_csv(os.path.join(test_result, f"scores_{timestamp}.csv"), index=False)
        print(f"Saved scores to {os.path.join(test_result, f'scores_{timestamp}.csv')}")



