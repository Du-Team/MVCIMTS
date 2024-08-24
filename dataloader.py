from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class GAIT(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'gait.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'gait.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'gait.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'gait.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'gait.mat')['X5'].astype(np.float32)
        data6 = scipy.io.loadmat(path + 'gait.mat')['X6'].astype(np.float32)
        labels = scipy.io.loadmat(path+'gait.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx]),
                torch.from_numpy(self.x6[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class ArticularyWordRecognition(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        data4 = np.loadtxt(path+'/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '4.txt', delimiter=',')[:,:-1].astype(np.float32)
        data5 = np.loadtxt(path+'/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '5.txt', delimiter=',')[:,:-1].astype(np.float32)
        data6 = np.loadtxt(path+'/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '6.txt', delimiter=',')[:,:-1].astype(np.float32)
        data7 = np.loadtxt(path + '/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '7.txt', delimiter=',')[:, :-1].astype(np.float32)
        data8 = np.loadtxt(path + '/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '8.txt', delimiter=',')[:, :-1].astype(np.float32)
        data9 = np.loadtxt(path + '/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '9.txt', delimiter=',')[:, :-1].astype(np.float32)
        labels = np.loadtxt(path+'/ArticularyWordRecognition/ArticularyWordRecognition_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.x7 = data7
        self.x8 = data8
        self.x9 = data9
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx]),
                torch.from_numpy(self.x6[idx]), torch.from_numpy(self.x7[idx]), torch.from_numpy(self.x8[idx]),
                torch.from_numpy(self.x9[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class BasicMotions(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/BasicMotions/BasicMotions_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/BasicMotions/BasicMotions_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/BasicMotions/BasicMotions_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        data4 = np.loadtxt(path+'/BasicMotions/BasicMotions_' + train_test + '4.txt', delimiter=',')[:,:-1].astype(np.float32)
        data5 = np.loadtxt(path+'/BasicMotions/BasicMotions_' + train_test + '5.txt', delimiter=',')[:,:-1].astype(np.float32)
        data6 = np.loadtxt(path+'/BasicMotions/BasicMotions_' + train_test + '6.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/BasicMotions/BasicMotions_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx]),
                torch.from_numpy(self.x6[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class Epilepsy(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/Epilepsy/Epilepsy_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/Epilepsy/Epilepsy_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/Epilepsy/Epilepsy_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/Epilepsy/Epilepsy_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()
class EthanolConcentration(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/EthanolConcentration/EthanolConcentration_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/EthanolConcentration/EthanolConcentration_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/EthanolConcentration/EthanolConcentration_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/EthanolConcentration/EthanolConcentration_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class HandMovementDirection(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/HandMovementDirection/HandMovementDirection_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/HandMovementDirection/HandMovementDirection_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/HandMovementDirection/HandMovementDirection_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        data4 = np.loadtxt(path + '/HandMovementDirection/HandMovementDirection_' + train_test + '4.txt', delimiter=',')[:, :-1].astype(np.float32)
        data5 = np.loadtxt(path + '/HandMovementDirection/HandMovementDirection_' + train_test + '5.txt', delimiter=',')[:, :-1].astype(np.float32)
        data6 = np.loadtxt(path + '/HandMovementDirection/HandMovementDirection_' + train_test + '6.txt', delimiter=',')[:, :-1].astype(np.float32)
        data7 = np.loadtxt(path + '/HandMovementDirection/HandMovementDirection_' + train_test + '7.txt', delimiter=',')[:, :-1].astype(np.float32)
        data8 = np.loadtxt(path + '/HandMovementDirection/HandMovementDirection_' + train_test + '8.txt', delimiter=',')[:, :-1].astype(np.float32)
        data9 = np.loadtxt(path + '/HandMovementDirection/HandMovementDirection_' + train_test + '9.txt', delimiter=',')[:, :-1].astype(np.float32)
        data10 = np.loadtxt(path + '/HandMovementDirection/HandMovementDirection_' + train_test + '10.txt', delimiter=',')[:, :-1].astype(np.float32)
        labels = np.loadtxt(path+'/HandMovementDirection/HandMovementDirection_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.x7 = data7
        self.x8 = data8
        self.x9 = data9
        self.x10 = data10
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx])
                , torch.from_numpy(self.x6[idx]), torch.from_numpy(self.x7[idx]), torch.from_numpy(self.x8[idx])
                , torch.from_numpy(self.x9[idx]), torch.from_numpy(self.x10[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class LP1(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/LP/LP1/1.txt', delimiter=' ').astype(np.float32)
        data2 = np.loadtxt(path+'/LP/LP1/2.txt', delimiter=' ').astype(np.float32)
        data3 = np.loadtxt(path+'/LP/LP1/3.txt', delimiter=' ').astype(np.float32)
        data4 = np.loadtxt(path + '/LP/LP1/4.txt', delimiter=' ').astype(np.float32)
        data5 = np.loadtxt(path + '/LP/LP1/5.txt', delimiter=' ').astype(np.float32)
        data6 = np.loadtxt(path + '/LP/LP1/6.txt', delimiter=' ').astype(np.float32)
        labels = np.loadtxt(path+'/LP/LP1/label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx])
                , torch.from_numpy(self.x6[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class LP4(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/LP/LP4/1.txt', delimiter=' ').astype(np.float32)
        data2 = np.loadtxt(path+'/LP/LP4/2.txt', delimiter=' ').astype(np.float32)
        data3 = np.loadtxt(path+'/LP/LP4/3.txt', delimiter=' ').astype(np.float32)
        data4 = np.loadtxt(path + '/LP/LP4/4.txt', delimiter=' ').astype(np.float32)
        data5 = np.loadtxt(path + '/LP/LP4/5.txt', delimiter=' ').astype(np.float32)
        data6 = np.loadtxt(path + '/LP/LP4/6.txt', delimiter=' ').astype(np.float32)
        labels = np.loadtxt(path+'/LP/LP4/label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx])
                , torch.from_numpy(self.x6[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class LP5(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/LP/LP5/1.txt', delimiter=' ').astype(np.float32)
        data2 = np.loadtxt(path+'/LP/LP5/2.txt', delimiter=' ').astype(np.float32)
        data3 = np.loadtxt(path+'/LP/LP5/3.txt', delimiter=' ').astype(np.float32)
        data4 = np.loadtxt(path + '/LP/LP5/4.txt', delimiter=' ').astype(np.float32)
        data5 = np.loadtxt(path + '/LP/LP5/5.txt', delimiter=' ').astype(np.float32)
        data6 = np.loadtxt(path + '/LP/LP5/6.txt', delimiter=' ').astype(np.float32)
        labels = np.loadtxt(path+'/LP/LP5/label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx])
                , torch.from_numpy(self.x6[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class PenDigits(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/PenDigits/PenDigits_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/PenDigits/PenDigits_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/PenDigits/PenDigits_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class Handwriting(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/Handwriting/Handwriting_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/Handwriting/Handwriting_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path + '/Handwriting/Handwriting_' + train_test + '3.txt', delimiter=',')[:, :-1].astype(np.float32)
        labels = np.loadtxt(path+'/Handwriting/Handwriting_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class RacketSports(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/RacketSports/RacketSports_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/RacketSports/RacketSports_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path + '/RacketSports/RacketSports_' + train_test + '3.txt', delimiter=',')[:, :-1].astype(np.float32)
        data4 = np.loadtxt(path + '/RacketSports/RacketSports_' + train_test + '4.txt', delimiter=',')[:, :-1].astype(np.float32)
        data5 = np.loadtxt(path + '/RacketSports/RacketSports_' + train_test + '5.txt', delimiter=',')[:, :-1].astype(np.float32)
        data6 = np.loadtxt(path + '/RacketSports/RacketSports_' + train_test + '6.txt', delimiter=',')[:, :-1].astype(np.float32)
        labels = np.loadtxt(path+'/RacketSports/RacketSports_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]),
                torch.from_numpy(self.x5[idx]), torch.from_numpy(self.x6[idx])], torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx)).long()

class Libras(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/Libras/Libras_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/Libras/Libras_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/Libras/Libras_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class UWaveGestureLibrary(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/UWaveGestureLibrary/UWaveGestureLibrary_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/UWaveGestureLibrary/UWaveGestureLibrary_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/UWaveGestureLibrary/UWaveGestureLibrary_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/UWaveGestureLibrary/UWaveGestureLibrary_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class LSST(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/LSST/LSST_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/LSST/LSST_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/LSST/LSST_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        data4 = np.loadtxt(path + '/LSST/LSST_' + train_test + '4.txt', delimiter=',')[:, :-1].astype(np.float32)
        data5 = np.loadtxt(path + '/LSST/LSST_' + train_test + '5.txt', delimiter=',')[:, :-1].astype(np.float32)
        data6 = np.loadtxt(path + '/LSST/LSST_' + train_test + '6.txt', delimiter=',')[:, :-1].astype(np.float32)
        labels = np.loadtxt(path+'/LSST/LSST_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]),
                torch.from_numpy(self.x5[idx]), torch.from_numpy(self.x6[idx])], \
               torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class StandWalkJump(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/StandWalkJump/StandWalkJump_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/StandWalkJump/StandWalkJump_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/StandWalkJump/StandWalkJump_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        data4 = np.loadtxt(path + '/StandWalkJump/StandWalkJump_' + train_test + '4.txt', delimiter=',')[:, :-1].astype(np.float32)
        labels = np.loadtxt(path+'/StandWalkJump/StandWalkJump_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx])], \
               torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class SelfRegulationSCP1(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/SelfRegulationSCP1/SelfRegulationSCP1_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/SelfRegulationSCP1/SelfRegulationSCP1_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/SelfRegulationSCP1/SelfRegulationSCP1_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        data4 = np.loadtxt(path + '/SelfRegulationSCP1/SelfRegulationSCP1_' + train_test + '4.txt', delimiter=',')[:, :-1].astype(np.float32)
        data5 = np.loadtxt(path + '/SelfRegulationSCP1/SelfRegulationSCP1_' + train_test + '5.txt', delimiter=',')[:,:-1].astype(np.float32)
        data6 = np.loadtxt(path + '/SelfRegulationSCP1/SelfRegulationSCP1_' + train_test + '6.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/SelfRegulationSCP1/SelfRegulationSCP1_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx]),
                torch.from_numpy(self.x6[idx])], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class SelfRegulationSCP2(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/SelfRegulationSCP2/SelfRegulationSCP2_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/SelfRegulationSCP2/SelfRegulationSCP2_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path+'/SelfRegulationSCP2/SelfRegulationSCP2_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        data4 = np.loadtxt(path + '/SelfRegulationSCP2/SelfRegulationSCP2_' + train_test + '4.txt', delimiter=',')[:, :-1].astype(np.float32)
        data5 = np.loadtxt(path + '/SelfRegulationSCP2/SelfRegulationSCP2_' + train_test + '5.txt', delimiter=',')[:,:-1].astype(np.float32)
        data6 = np.loadtxt(path + '/SelfRegulationSCP2/SelfRegulationSCP2_' + train_test + '6.txt', delimiter=',')[:,:-1].astype(np.float32)
        data7 = np.loadtxt(path + '/SelfRegulationSCP2/SelfRegulationSCP2_' + train_test + '7.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/SelfRegulationSCP2/SelfRegulationSCP2_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.x7 = data7
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx]),
                torch.from_numpy(self.x6[idx]), torch.from_numpy(self.x7[idx])], \
               torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class AtrialFibrillation(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/AtrialFibrillation/AtrialFibrillation_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/AtrialFibrillation/AtrialFibrillation_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/AtrialFibrillation/AtrialFibrillation_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], \
               torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class FingerMovements(Dataset):
    def __init__(self, path, train_test):
        data1 = np.loadtxt(path+'/FingerMovements/FingerMovements_' + train_test + '1.txt', delimiter=',')[:,:-1].astype(np.float32)
        data2 = np.loadtxt(path+'/FingerMovements/FingerMovements_' + train_test + '2.txt', delimiter=',')[:,:-1].astype(np.float32)
        data3 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '3.txt', delimiter=',')[:,:-1].astype(np.float32)
        data4 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '4.txt', delimiter=',')[:,:-1].astype(np.float32)
        data5 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '5.txt', delimiter=',')[:,:-1].astype(np.float32)
        data6 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '6.txt', delimiter=',')[:,:-1].astype(np.float32)
        data7 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '7.txt', delimiter=',')[:,:-1].astype(np.float32)
        data8 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '8.txt', delimiter=',')[:,:-1].astype(np.float32)
        data9 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '9.txt', delimiter=',')[:,:-1].astype(np.float32)
        data10 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '10.txt', delimiter=',')[:,:-1].astype(np.float32)
        data11 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '11.txt', delimiter=',')[:,:-1].astype(np.float32)
        data12 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '12.txt', delimiter=',')[:,:-1].astype(np.float32)
        data13 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '13.txt', delimiter=',')[:,:-1].astype(np.float32)
        data14 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '14.txt', delimiter=',')[:,:-1].astype(np.float32)
        data15 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '15.txt', delimiter=',')[:,:-1].astype(np.float32)
        data16 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '16.txt', delimiter=',')[:,:-1].astype(np.float32)
        data17 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '17.txt', delimiter=',')[:,:-1].astype(np.float32)
        data18 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '18.txt', delimiter=',')[:,:-1].astype(np.float32)
        data19 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '19.txt', delimiter=',')[:,:-1].astype(np.float32)
        data20 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '20.txt', delimiter=',')[:,:-1].astype(np.float32)
        data21 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '21.txt', delimiter=',')[:,:-1].astype(np.float32)
        data22 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '22.txt', delimiter=',')[:,:-1].astype(np.float32)
        data23 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '23.txt', delimiter=',')[:,:-1].astype(np.float32)
        data24 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '24.txt', delimiter=',')[:,:-1].astype(np.float32)
        data25 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '25.txt', delimiter=',')[:,:-1].astype(np.float32)
        data26 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '26.txt', delimiter=',')[:,:-1].astype(np.float32)
        data27 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '27.txt', delimiter=',')[:,:-1].astype(np.float32)
        data28 = np.loadtxt(path + '/FingerMovements/FingerMovements_' + train_test + '28.txt', delimiter=',')[:,:-1].astype(np.float32)
        labels = np.loadtxt(path+'/FingerMovements/FingerMovements_' + train_test + '_label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.x7 = data7
        self.x8 = data8
        self.x9 = data9
        self.x10 = data10
        self.x11 = data11
        self.x12 = data12
        self.x13 = data13
        self.x14 = data14
        self.x15 = data15
        self.x16 = data16
        self.x17 = data17
        self.x18 = data18
        self.x19 = data19
        self.x20 = data20
        self.x21 = data21
        self.x22 = data22
        self.x23 = data23
        self.x24 = data24
        self.x25 = data25
        self.x26 = data26
        self.x27 = data27
        self.x28 = data28
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx]),  torch.from_numpy(self.x4[idx]),
                torch.from_numpy(self.x5[idx]),  torch.from_numpy(self.x6[idx]),
                torch.from_numpy(self.x7[idx]),  torch.from_numpy(self.x8[idx]),
                torch.from_numpy(self.x9[idx]),  torch.from_numpy(self.x10[idx]),
                torch.from_numpy(self.x11[idx]),  torch.from_numpy(self.x12[idx]),
                torch.from_numpy(self.x13[idx]),  torch.from_numpy(self.x14[idx]),
                torch.from_numpy(self.x15[idx]),  torch.from_numpy(self.x16[idx]),
                torch.from_numpy(self.x17[idx]),  torch.from_numpy(self.x18[idx]),
                torch.from_numpy(self.x19[idx]),  torch.from_numpy(self.x20[idx]),
                torch.from_numpy(self.x21[idx]),  torch.from_numpy(self.x22[idx]),
                torch.from_numpy(self.x23[idx]), torch.from_numpy(self.x24[idx]),
                torch.from_numpy(self.x25[idx]), torch.from_numpy(self.x26[idx]),
                torch.from_numpy(self.x27[idx]), torch.from_numpy(self.x28[idx]),], \
               torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class II_Ia(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        data1 = scaler.fit_transform(np.loadtxt(path+'/II_Ia_data_' + '1.txt', delimiter=',')).astype(np.float32)
        data2 = scaler.fit_transform(np.loadtxt(path+'/II_Ia_data_' + '2.txt', delimiter=',')).astype(np.float32)
        data3 = scaler.fit_transform(np.loadtxt(path + '/II_Ia_data_' + '3.txt', delimiter=',')).astype(np.float32)
        data4 = scaler.fit_transform(np.loadtxt(path + '/II_Ia_data_' + '4.txt', delimiter=',')).astype(np.float32)
        data5 = scaler.fit_transform(np.loadtxt(path + '/II_Ia_data_' + '5.txt', delimiter=',')).astype(np.float32)
        data6 = scaler.fit_transform(np.loadtxt(path + '/II_Ia_data_' + '6.txt', delimiter=',')).astype(np.float32)
        labels = np.loadtxt(path+'/II_Ia_data_' + 'label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]),
                torch.from_numpy(self.x5[idx]),torch.from_numpy(self.x6[idx])], \
               torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()
class II_Ib(Dataset):
    def __init__(self, path):
        data1 = np.loadtxt(path+'/II_Ib_data_' + '1.txt', delimiter=',').astype(np.float32)
        data2 = np.loadtxt(path+'/II_Ib_data_' + '2.txt', delimiter=',').astype(np.float32)
        data3 = np.loadtxt(path + '/II_Ib_data_' + '3.txt', delimiter=',').astype(np.float32)
        data4 = np.loadtxt(path + '/II_Ib_data_' + '4.txt', delimiter=',').astype(np.float32)
        data5 = np.loadtxt(path + '/II_Ib_data_' + '5.txt', delimiter=',').astype(np.float32)
        data6 = np.loadtxt(path + '/II_Ib_data_' + '6.txt', delimiter=',').astype(np.float32)
        data7 = np.loadtxt(path + '/II_Ib_data_' + '7.txt', delimiter=',').astype(np.float32)
        labels = np.loadtxt(path+'/II_Ib_data_' + 'label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.x7 = data7
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]),
                torch.from_numpy(self.x5[idx]),torch.from_numpy(self.x6[idx]),
                torch.from_numpy(self.x7[idx])], \
               torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class III_V_s2(Dataset):
    def __init__(self, path):
        data1 = np.loadtxt(path+'/III_V_s2_data_' + '1.txt', delimiter=',').astype(np.float32)
        data2 = np.loadtxt(path+'/III_V_s2_data_' + '2.txt', delimiter=',').astype(np.float32)
        data3 = np.loadtxt(path + '/III_V_s2_data_' + '3.txt', delimiter=',').astype(np.float32)
        data4 = np.loadtxt(path + '/III_V_s2_data_' + '4.txt', delimiter=',').astype(np.float32)
        data5 = np.loadtxt(path + '/III_V_s2_data_' + '5.txt', delimiter=',').astype(np.float32)
        data6 = np.loadtxt(path + '/III_V_s2_data_' + '6.txt', delimiter=',').astype(np.float32)
        data7 = np.loadtxt(path + '/III_V_s2_data_' + '7.txt', delimiter=',').astype(np.float32)
        data8 = np.loadtxt(path + '/III_V_s2_data_' + '8.txt', delimiter=',').astype(np.float32)
        labels = np.loadtxt(path+'/III_V_s2_data_' + 'label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.x7 = data7
        self.x8 = data8
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]),
                torch.from_numpy(self.x5[idx]),torch.from_numpy(self.x6[idx]),
                torch.from_numpy(self.x7[idx]),torch.from_numpy(self.x8[idx])], \
               torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class IV_2b_s2(Dataset):
    def __init__(self, path):
        data1 = np.loadtxt(path+'/IV_2b_s2_data_' + '1.txt', delimiter=',').astype(np.float32)
        data2 = np.loadtxt(path+'/IV_2b_s2_data_' + '2.txt', delimiter=',').astype(np.float32)
        data3 = np.loadtxt(path + '/IV_2b_s2_data_' + '3.txt', delimiter=',').astype(np.float32)
        labels = np.loadtxt(path+'/IV_2b_s2_data_' + 'label.txt')
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx])], \
               torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()

class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset,train_test):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "gait":
        dataset = GAIT('./data/')
        dims = [1010, 1010, 1010, 1010, 1010, 1010]
        view = 6
        data_size = 30
        class_num = 3
    elif dataset == "ArticularyWordRecognition":
        dataset = ArticularyWordRecognition('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        view = 9
        data_size = len(dataset)
        class_num = 25
    elif dataset == "BasicMotions":
        dataset = BasicMotions('./data/MTS',train_test)
        dims = [1, 1, 1, 1, 1, 1]
        view = 6
        data_size = len(dataset)
        class_num = 4
    elif dataset == "Epilepsy":
        dataset = Epilepsy('./data/MTS',train_test)
        dims = [1, 1, 1]
        view = 3
        data_size = len(dataset)
        class_num = 4
    elif dataset == "EthanolConcentration":
        dataset = EthanolConcentration('./data/MTS', train_test)
        dims = [1, 1, 1]
        view = 3
        data_size = len(dataset)
        class_num = 4
    elif dataset == "HandMovementDirection":
        dataset = HandMovementDirection('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        view = 10
        data_size = len(dataset)
        class_num = 4
    elif dataset == "PenDigits":
        dataset = PenDigits('./data/MTS', train_test)
        dims = [1, 1]
        view = 2
        data_size = len(dataset)
        class_num = 10
    elif dataset == "Handwriting":
        dataset = Handwriting('./data/MTS', train_test)
        dims = [1, 1, 1]
        view = 3
        data_size = len(dataset)
        class_num = 26
    elif dataset == "RacketSports":
        dataset = RacketSports('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1]
        view = 6
        data_size = len(dataset)
        class_num = 4
    elif dataset == "Libras":
        dataset = Libras('./data/MTS', train_test)
        dims = [1, 1]
        view = 2
        data_size = len(dataset)
        class_num = 15
    elif dataset == "UWaveGestureLibrary":
        dataset = UWaveGestureLibrary('./data/MTS', train_test)
        dims = [1, 1, 1]
        view = 3
        data_size = len(dataset)
        class_num = 8
    elif dataset == "LSST":
        dataset = LSST('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1]
        view = 6
        data_size = len(dataset)
        class_num = 14
    elif dataset == "StandWalkJump":
        dataset = StandWalkJump('./data/MTS', train_test)
        dims = [1, 1, 1, 1]
        view = 4
        data_size = len(dataset)
        class_num = 3
    elif dataset == "SelfRegulationSCP1":
        dataset = SelfRegulationSCP1('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1]
        view = 6
        data_size = len(dataset)
        class_num = 2
    elif dataset == "SelfRegulationSCP2":
        dataset = SelfRegulationSCP2('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1, 1]
        view = 7
        data_size = len(dataset)
        class_num = 2
    elif dataset == "AtrialFibrillation":
        dataset = AtrialFibrillation('./data/MTS', train_test)
        dims = [1, 1]
        view = 2
        data_size = len(dataset)
        class_num = 3
    elif dataset == "FingerMovements":
        dataset = FingerMovements('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        view = 28
        data_size = len(dataset)
        class_num = 2
    elif dataset == "LP1":
        dataset = LP1('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1]
        view = 6
        data_size = len(dataset)
        class_num = 4
    elif dataset == "LP4":
        dataset = LP4('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1]
        view = 6
        data_size = len(dataset)
        class_num = 3
    elif dataset == "LP5":
        dataset = LP5('./data/MTS', train_test)
        dims = [1, 1, 1, 1, 1, 1]
        view = 6
        data_size = len(dataset)
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "II_Ia_data":
        dataset = II_Ia('./data/EEG-data-and-descriptions-master')
        dims = [1, 1, 1, 1, 1, 1]
        view = 6
        data_size = 268
        class_num = 2
    elif dataset == "II_Ib_data":
        dataset = II_Ib('./data/EEG-data-and-descriptions-master')
        dims = [1, 1, 1, 1, 1, 1, 1]
        view = 7
        data_size = 200
        class_num = 2
    elif dataset == "III_V_s2_data":
        dataset = III_V_s2('./data/EEG-data-and-descriptions-master')
        dims = [1, 1, 1, 1, 1, 1, 1, 1]
        view = 8
        data_size = 3472
        class_num = 3
    elif dataset == "IV_2b_s2_data":
        dataset = IV_2b_s2('./data/EEG-data-and-descriptions-master')
        dims = [1, 1, 1]
        view = 3
        data_size = 120
        class_num = 2
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
