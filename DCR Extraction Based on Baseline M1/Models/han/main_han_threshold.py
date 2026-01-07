from datetime import datetime
import pandas as pd
import copy
import os.path
import random
import torch
from sklearn.metrics import f1_score
from torch.ao.nn.quantized.functional import threshold
from tqdm import tqdm
import kmeans
from sklearn.metrics import silhouette_score
from Models.model1 import Model
from utils import EarlyStopping, load_test_data

def check_clusters(cluster):
    unique_clusters = torch.unique(cluster)
    if len(unique_clusters) < 2:
        print(f"聚类结果中只有一个簇，无法计算轮廓系数。簇的数量: {len(unique_clusters)}")
        return False
    return True

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

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
scores_list = []
num_clusters = 2


def main(args, data_path, output_folder):
    batch_size = 100
    batch_total = 10000


    g, features, positive_samples, negative_samples, files,all_data = load_test_data(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    model = Model(meta_paths=[['fa', 'af']],
                  embedding_size=71936,
                  hidden_size=args['hidden_units'],
                  out_size=4,
                  num_heads=args['num_heads'],
                  dropout=args['dropout']).to(device)

    model = model.to(device)

    best_score=0
    best_model=None

    g = g.to(device)
    features = features.to(device)

    stopper = EarlyStopping(str(os.path.basename(data_path)).split('.')[0],patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                 weight_decay=args['weight_decay'])

    import random

    test_samples=random.sample(range(features.shape[0]),int(features.shape[0]/3))
    _test_samples=set(test_samples)
    _positive_samples=[]
    _negative_samples=[]

    train_samples=list(set(range(features.shape[0]))-_test_samples)


    for i in positive_samples:
        if i[0] not in _test_samples and i[1] not in _test_samples:
            _positive_samples.append(i)

    for i in negative_samples:
        if i[0] not in _test_samples and i[1] not in _test_samples:
            _negative_samples.append(i)


    print(positive_samples.__len__(), _positive_samples.__len__())
    print(negative_samples.__len__(), _negative_samples.__len__())

    positive_samples=_positive_samples
    negative_samples=_negative_samples

    valid_samples=test_samples[0:int(test_samples.__len__()/2)]
    test_samples=test_samples[int(test_samples.__len__()/2):]


    batch_total=int(min(positive_samples.__len__(),negative_samples.__len__(),batch_total)/batch_size)*batch_size
    best_epoch=0
    # for epoch in range(50):
    for epoch in range(50):
        model.train()
        random.shuffle(positive_samples)
        random.shuffle(negative_samples)


        total_loss=0
        for i in tqdm(range(int(batch_total/batch_size))):
            output = model(g,features)
            p = positive_samples[i * batch_size:(i + 1) * batch_size]
            n = negative_samples[i * batch_size:(i + 1) * batch_size]
            a, b, c,d = [], [], [],[]
            for i in p:
                a.append( i[0])
                b .append( i[1])
            for i in n:
                c.append( i[0])
                d.append(i[1])
            l1=model._forward_train_positive(output,a,b)
            l2=model._forward_train_negative(output,c,d)
            loss=l1+l2
            # loss.requires_grad_(True)
            optimizer.zero_grad()
            total_loss+=loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        output = model(g, features)

        _output=torch.tensor([output[i].cpu().detach().numpy() for i in train_samples]).to(device)
        cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
        # score1 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
        if check_clusters(cluster):
            score1 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
        else:
            score1 = None

        _output=torch.tensor([output[i].cpu().detach().numpy() for i in valid_samples]).to(device)
        cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
        # score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
        if check_clusters(cluster):
            score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
        else:
            score2 = None
        print("EPOCH:{}  score train:{} score valid:{}  average_loss:".format(epoch, score1,score2), total_loss / (batch_total / batch_size))

        # if score2 > best_score:
        #     best_score = score2
        #     best_model = copy.deepcopy(Models)
        #     best_epoch = epoch
        #     print(f"New best Models found at epoch {epoch} with validation score: {score2}")
        if score2 is not None and best_score is not None:
            if score2 > best_score:
                best_score = score2
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                print(f"New best model found at epoch {epoch} with validation score: {score2}")
        else:
            print(
                f"Skipping comparison because score2 or best_score is None. score2: {score2}, best_score: {best_score}")
        if epoch - best_epoch > args['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
        # stopper.save_checkpoint(Models)


    best_model.eval()
    output = best_model(g, features)

    _output = output
    cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
    # score0 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
    if check_clusters(cluster):
        score0 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
    else:
        score0 = None

    _output = torch.tensor([output[i].cpu().detach().numpy() for i in valid_samples]).to(device)
    cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
    # score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
    if check_clusters(cluster):
        score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
    else:
        score2 = None
    _output = torch.tensor([output[i].cpu().detach().numpy() for i in test_samples]).to(device)
    cluster, center = kmeans.kmeans(_output, num_clusters, device=device)
    # score3 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
    if check_clusters(cluster):
        score3 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())
    else:
        score3 = None

    print("best_epoch:{}   score total:{}  score test:{}   score valid:{} ".format(best_epoch,score0,score3,score2))
    # Save the best Models to the output folder

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

    test_result = '/home/user/wyh/DC_contribution/finally_project/code/HAN/unsupervised_output/HAN/result/mlkit'
    output_folder ='/home/user/wyh/DC_contribution/finally_project/code/HAN/unsupervised_output/HAN/save_model/mlkit'
    os.makedirs(test_result, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Load all .pkl files in the folder and train Models
    data_folder = '/home/user/wyh/DC_contribution/finally_project/code/RFmaster/pkl_files/mlkit_unlabel/Experiment_20250221-191646_0_7'
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


