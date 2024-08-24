import torch.nn as nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_size1=128, hidden_size2=256):
        super(Encoder, self).__init__()
        #self.encoder = nn.Sequential(
            # nn.Linear(input_dim, 500),
            # nn.ReLU(),
            # nn.Linear(500, 500),
            # nn.ReLU(),
            # nn.Linear(500, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, feature_dim),
        #)
        self.encoder_gru_layer1 = nn.GRU(input_dim, hidden_size1, num_layers=1, batch_first=True, dropout=0.2)
        self.encoder_gru_layer2 = nn.GRU(hidden_size1, hidden_size2, num_layers=1, batch_first=True, dropout=0.2)
        self.encoder_linear_layer = nn.Linear(hidden_size2, feature_dim)

    def forward(self, x):

       # return self.encoder(x)
       x = x.reshape(x.shape[0], x.shape[1], 1)
       output1, _ = self.encoder_gru_layer1(x)
       output1 = F.relu(output1)

       output2, _ = self.encoder_gru_layer2(output1)
       output2 = F.relu(output2)

       # 取最后一个时间步的输出作为编码特征
       last_output = output2[:, -1, :]

       # 通过线性层进行映射
       feature = self.encoder_linear_layer(last_output)

       return feature


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_size1=128, hidden_size2=256):
        super(Decoder, self).__init__()
        #self.decoder = nn.Sequential(
            # nn.Linear(feature_dim, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 500),
            # nn.ReLU(),
            # nn.Linear(500, 500),
            # nn.ReLU(),
            # nn.Linear(500, input_dim)
        #)
        self.decoder_gru_layer1 = nn.GRU(feature_dim, hidden_size2, num_layers=1, batch_first=True, dropout=0.2)
        self.decoder_gru_layer2 = nn.GRU(hidden_size2, hidden_size1, num_layers=1, batch_first=True, dropout=0.2)
        self.decoder_linear_layer = nn.Linear(hidden_size1, input_dim)

    def forward(self, x):
        #return self.decoder(x)

        output1, _ = self.decoder_gru_layer1(x)

        output2, _ = self.decoder_gru_layer2(output1)

        recon_output = self.decoder_linear_layer(output2)

        return recon_output

class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,prediction_dim,activation='relu',batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self.depth = len(prediction_dim) - 1
        self.activation = activation
        self.prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self.depth):
            encoder_layers.append(
                nn.Linear(self.prediction_dim[i], self.prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self.prediction_dim[i + 1]))
            if self.activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self.activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self.activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self.activation)
        self.encoder = nn.Sequential(*encoder_layers)


        decoder_layers = []
        for i in range(self.depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self.prediction_dim[i], self.prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self.prediction_dim[i - 1]))

                if self.activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self.activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self.activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self.activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, mask, device, config):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.view1_view2 = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
            self.dims_view = [feature_dim] + config['Prediction']['view{}'.format(v)]
            self.view1_view2.append(Prediction(self.dims_view).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view
        self.feature_dim = feature_dim
        self.mask = mask
        self.config = config
        self.device = device

    def forward(self, xs, index=None, iftest=False):   #全提取
        hs = []
        qs = []
        xrs = []
        zs = []
        pred_view = []
        if iftest:
            for v in range(self.view):
                idx_eval = self.mask[index, v] == 1
                missing_idx_eval = self.mask[index, v] == 0
                x = xs[v]
                z = self.encoders[v](x[idx_eval])
                latent = torch.zeros(x.shape[0], self.feature_dim).to(self.device)
                missing_z = self.encoders[v](x[missing_idx_eval])
                if missing_z.shape[0] == 1:
                    missing_z = torch.cat((missing_z, missing_z), dim=0)
                    missing_eval, _ = self.view1_view2[v](missing_z)
                    missing_eval = missing_eval[0]
                else:
                    missing_eval, _ = self.view1_view2[v](missing_z)
                latent[idx_eval] = z
                latent[missing_idx_eval] = missing_eval

                h = normalize(self.feature_contrastive_module(latent), dim=1)  # 高级特征
                q = self.label_contrastive_module(latent)  # 语义特征
                xr = self.decoders[v](latent)  # 重构数据
                hs.append(h)  # 高级特征累加
                zs.append(latent)  # 低级特征累加
                qs.append(q)  # 语义特征累加
                xrs.append(xr)  # 重构数据累加
                pred_view.append(latent)

        else:
            for v in range(self.view):
                x = xs[v]
                z = self.encoders[v](x)   #低级特征
                h = normalize(self.feature_contrastive_module(z), dim=1)   #高级特征
                q = self.label_contrastive_module(z)   #语义特征
                #self.dims_view = [self.feature_dim] + self.config['Prediction']['view{}'.format(v)]
                #self.view1_view2 = Prediction(self.dims_view)
                img2txt, _ = self.view1_view2[v](z)
                xr = self.decoders[v](z)  #重构数据
                hs.append(h)   #高级特征累加
                zs.append(z)   #低级特征累加
                qs.append(q)   #语义特征累加
                xrs.append(xr)  #重构数据累加
                pred_view.append(img2txt)
        return hs, qs, xrs, zs, pred_view

    def forward_plot(self, xs, iftest=False):
        zs = []
        hs = []
        if iftest:
            for v in range(self.view):
                idx_eval = self.mask[:, v] == 1
                missing_idx_eval = self.mask[:, v] == 0
                x = xs[v]
                z = self.encoders[v](x[idx_eval])
                latent = torch.zeros(x.shape[0], self.feature_dim).to(self.device)
                missing_z = self.encoders[v](x[missing_idx_eval])
                if missing_z.shape[0] == 1:
                    missing_z = torch.cat((missing_z, missing_z), dim=0)
                    missing_eval, _ = self.view1_view2[v](missing_z)
                    missing_eval = missing_eval[0]
                else:
                    missing_eval, _ = self.view1_view2[v](missing_z)
                latent[idx_eval] = z
                latent[missing_idx_eval] = missing_eval
                zs.append(latent)
                h = self.feature_contrastive_module(latent)  # 高级特征
                hs.append(h)  # 高级特征累加

        else:
            for v in range(self.view):
                x = xs[v]
                z = self.encoders[v](x)   #低级特征
                zs.append(z)    #低级特征累加
                h = self.feature_contrastive_module(z)  #高级特征
                hs.append(h)  #高级特征累加
        return zs, hs

    def forward_cluster(self, xs, index=None, iftest=False):
        qs = []
        preds = []
        if iftest:
            for v in range(self.view):
                idx_eval = self.mask[index, v] == 1
                missing_idx_eval = self.mask[index, v] == 0
                x = xs[v]
                z = self.encoders[v](x[idx_eval])
                latent = torch.zeros(x.shape[0], self.feature_dim).to(self.device)
                missing_z = self.encoders[v](x[missing_idx_eval])
                if missing_z.shape[0] == 1:
                    missing_z = torch.cat((missing_z, missing_z), dim=0)
                    missing_eval, _ = self.view1_view2[v](missing_z)
                    missing_eval = missing_eval[0]
                else:
                    missing_eval, _ = self.view1_view2[v](missing_z)
                latent[idx_eval] = z
                latent[missing_idx_eval] = missing_eval

                q = self.label_contrastive_module(latent)
                pred = torch.argmax(q, dim=1)
                qs.append(q)
                preds.append(pred)

        else:
            for v in range(self.view):
                x = xs[v]
                z = self.encoders[v](x)
                q = self.label_contrastive_module(z)
                pred = torch.argmax(q, dim=1)
                qs.append(q)
                preds.append(pred)
        return qs, preds    #语义特征，预测标签