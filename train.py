import torch
import torch.nn.functional as F
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
from loss import Loss
from dataloader import load_data
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from configure import get_default_config
from get_mask import get_mask
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# batch_size = [2,4,8,16,32]
# feature_dim = [4,8,16,32,64,128,256,512]

Dataname = 'RacketSports'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument("--temperature_f", default=0.5)  #0.5【0.5，0.6，0.7，0.8，0.9，1】
parser.add_argument("--temperature_l", default=1)    #1 【0.4，0.5，0.6，0.7，0.8，0.9，1】
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=300)
parser.add_argument("--con_epochs", default=200)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=128)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "BDGP":
    args.con_epochs = 10
    seed = 10
if args.dataset == "gait":
    args.con_epochs = 10
    seed = 10
if args.dataset == "ArticularyWordRecognition":
    args.con_epochs = 200
    seed = 10
if args.dataset == "BasicMotions":
    args.con_epochs = 200
    seed = 10
if args.dataset == "Epilepsy":
    args.con_epochs = 200
    seed = 10
if args.dataset == "EthanolConcentration":
    args.con_epochs = 100
    seed = 10
if args.dataset == "HandMovementDirection":
    args.con_epochs = 200
    seed = 10
if args.dataset == "PenDigits":
    args.con_epochs = 250
    seed = 10
if args.dataset == "Handwriting":
    args.con_epochs = 50
    seed = 10
if args.dataset == "RacketSports":
    args.con_epochs = 200
    seed = 10
if args.dataset == "Libras":
    args.con_epochs = 200
    seed = 10
if args.dataset == "UWaveGestureLibrary":
    args.con_epochs = 200
    seed = 10
if args.dataset == "LSST":
    args.con_epochs = 200
    seed = 10
if args.dataset == "StandWalkJump":
    args.con_epochs = 200
    seed = 10
if args.dataset == "SelfRegulationSCP1":
    args.con_epochs = 200
    seed = 10
if args.dataset == "SelfRegulationSCP2":
    args.con_epochs = 100
    seed = 10
if args.dataset == "AtrialFibrillation":
    args.con_epochs = 200
    seed = 10
if args.dataset == "FingerMovements":
    args.con_epochs = 100
    seed = 10
if args.dataset == "LP1":
    args.con_epochs = 200
    seed = 10
if args.dataset == "LP4":
    args.con_epochs = 250
    seed = 10
if args.dataset == "LP5":
    args.con_epochs = 200
    seed = 10
if args.dataset == "II_Ia_data":
    args.con_epochs = 200
    seed = 10
if args.dataset == "II_Ib_data":
    args.con_epochs = 200
    seed = 10
if args.dataset == "III_V_s2_data":
    args.con_epochs = 200
    seed = 10
if args.dataset == "IV_2b_s2_data":
    args.con_epochs = 200
    seed = 10



config = get_default_config(args.dataset)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


dataset, dims, view, data_size, class_num = load_data(args.dataset, train_test='train')

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

mask = get_mask(view, len(dataset), config['training']['missing_rate'])

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, zs, pred_view = model(xs)   #xrs重构数据
        recloss_list = []
        preloss_list = []
        for v in range(view):
            recloss_list.append(criterion(xs[v], xrs[v]))   #只重构损失
            preloss_list.append(criterion(pred_view[v], zs[v]))
        loss = sum(recloss_list) + sum(preloss_list) #所有视图的损失
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, pred_view = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))  #高级特征
                loss_list.append(criterion.forward_label(qs[v], qs[w]))   #语义特征
            loss_list.append(mse(xs[v], xrs[v]))
            loss_list.append(mse(pred_view[v],zs[v]))
        # loss = sum(loss_list)   #四种损失的和
        loss = 0.1*loss_list[0] + 0.1*loss_list[1] + loss_list[2] + 100*loss_list[3]
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    return tot_loss/len(data_loader)

# 在高级特征上用kmeans得到聚类标签
def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _, _ = model.forward(xs)  #hs高级特征

        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])
            print("hs的形状：",hs[v].shape)



    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(data_size, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)
    merged_features = np.hstack(hs)
    kmeans.fit(merged_features)
    centroids = kmeans.cluster_centers_
    labels = Pseudo_label
    visual(merged_features, centroids, labels)

    return new_pseudo_label


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y

def visual(x, centroids, labels):
    # 将质心添加到数据集中，并标记它们以便后续区分
    n_samples, n_features = x.shape
    n_clusters, n_features = centroids.shape
    colors = cm.get_cmap('viridis', n_clusters)

    is_complete = np.random.choice([True, False], size=n_samples, p=[0.5, 0.5])  # 70% 数据完整，30% 数据不完整
    all_data = np.vstack((x, centroids))
    is_centroid = np.zeros(all_data.shape[0], dtype=bool)
    is_centroid[-centroids.shape[0]:] = True

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=0)
    all_data_2d = tsne.fit_transform(all_data)

    # 分离出原始数据点和质心的低维表示
    X_2d = all_data_2d[:-centroids.shape[0], :]
    centroids_2d = all_data_2d[-centroids.shape[0]:, :]

    labels = labels.squeeze()

    for i in range(n_clusters):
        cluster_complete = X_2d[is_complete & (labels.numpy() == i),:]
        plt.scatter(cluster_complete[:,0], cluster_complete[:,1], c=colors(i),
                    marker='.')
        print("完整数据的形状:", cluster_complete.shape)

        # 绘制不完整数据叉号，使用不同的颜色表示不同的簇
    for i in range(n_clusters):
        cluster_incomplete = X_2d[~is_complete & (labels.numpy() == i),:]
        plt.scatter(cluster_incomplete[:,0], cluster_incomplete[:,1], c=colors(i),
                     marker='x', s=10)
        print("不完整数据的形状:", cluster_incomplete.shape)

    # # 绘制完整数据点
    # plt.scatter(X_2d[is_complete, 0], X_2d[is_complete, 1], c='red', label='Complete Data', marker='.')
    #
    # # 绘制不完整数据叉号
    # plt.scatter(X_2d[~is_complete, 0], X_2d[~is_complete, 1], c='green', label='Incomplete Data', marker='x')

    # 绘制质心星号
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', marker='*')

    # 不显示刻度线
    plt.xticks([])  # X轴刻度线为空
    plt.yticks([])  # Y轴刻度线为空

    # 保存图像为PDF
    #plt.savefig('RacketSports_0.9.pdf', format='pdf')

    # 显示图像
    plt.show()




def fine_tuning(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.train()
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, _, _ = model(xs)  #语义特征
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return tot_loss / len(data_loader)

#
def draw_loss(y_losses, ri_scores, nmi_scores, acc_scores):

    # y_losses = list(i/1000 for i in y_losses)
    # plt.figure(figsize=(6, 4))
    # x_epochs = list(range(1, len(y_losses) + 1))
    # plt.plot(x_epochs, y_losses)
    # plt.xticks(list(range(0, len(y_losses) + 1,50)))
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # #plt.savefig('./fig/Epilepsy1_loss.pdf', format='pdf')
    # plt.show()

    plt.figure(figsize=(10, 6))
    x_epochs = list(range(1, len(y_losses) + 1)) #np.arange(len(y_losses))
    ax1 = plt.gca()
    line_ri, = ax1.plot(x_epochs, ri_scores, color='green', linestyle='-', marker='o')
    line_nmi, = ax1.plot(x_epochs, nmi_scores, color='red', linestyle='-', marker='s')
    line_acc, = ax1.plot(x_epochs, acc_scores, color='purple', linestyle='-', marker='^')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Clustering Performance')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0.0, 1.0)
    ax1.yaxis.set_ticks(np.arange(0.0, 1.1, 0.2))

    # 创建第二个y轴（右侧），用于损失值
    ax2 = ax1.twinx()
    line_loss, = ax2.plot(x_epochs, y_losses, label='Loss', color='blue', linestyle='-', marker='v')
    ax2.set_ylabel('Loss Value')
    ax2.tick_params(axis='y')
    ax2.set_ylim(min(y_losses) * 0.9, max(y_losses) * 1.1)

    # 合并图例
    lines = [line_ri, line_nmi, line_acc, line_loss]
    labels = ['RI', 'NMI', 'ACC', 'Loss']
    ax1.legend(lines, labels, loc='upper right', ncol=1, bbox_to_anchor=(1, 1), frameon=True)

    # plt.xticks(x_epochs)
    plt.xticks(list(range(0, len(y_losses) + 1,50)))  # 这里的 [::2] 表示每隔一个元素取一个
    plt.grid(True)
    plt.show()


ris = []
nmis = []
aris = []
accs = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
y_losses = []
ri_score = []
nmi_score = []
acc_score = []
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, mask, device, config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    start_time = time.time()
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_loss = contrastive_train(epoch)
        y_losses.append(contrastive_loss)
        if epoch == args.mse_epochs + args.con_epochs:
            ri, nmi, ari, acc, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
        epoch += 1
    #draw_loss(y_losses, ri_score, nmi_score, acc_score)
    new_pseudo_label = make_pseudo_label(model, device)
    while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
        tune_loss = fine_tuning(epoch, new_pseudo_label)
        #y_losses.append(tune_loss)
        if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
            ri, nmi, ari, acc, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
            state = model.state_dict()
            #torch.save(state, './models/' + args.dataset + '.pth')
            print('Saving..')
            ris.append(ri)
            nmis.append(nmi)
            aris.append(ari)
            accs.append(acc)
            purs.append(pur)
        epoch += 1
    end_time = time.time()
    run_time = end_time - start_time
    print("运行时间:",run_time)

