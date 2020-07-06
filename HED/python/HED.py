import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import random
random.seed(113)


def ED_method(PS, TH):
    PD_E = []
    m, n = PS.shape
    for i in range(m):
        H0 = 0
        H1 = 0
        data = PS[i, :]
        #  compare to the threshold:
        for j in range(n):
            if data[j] > TH:
                H1 = H1 + 1
            else:
                H0 = H0 + 1
        Pd = H1 / n
        PD_E.append(Pd)
    PD_E = np.array(PD_E)
    # plt.plot(PD_E, 'r--*', label='ED'), plt.legend(fontsize=18)
    # plt.show()
    return PD_E


def KNN_method(PS, PN, K):
    PD_K = []
    PF_K = []
    random.seed(113)
    m, n = PS.shape
    for i in range(m):
        data1 = PS[i, :]  # positive samples
        label1 = np.ones(len(data1))  # positive samples' labels
        label2 = np.zeros(len(PN))  # negative samples' labels
        labels = np.hstack((label1, label2))  # all labels
        index1 = np.arange(0, len(labels), 1)
        random.shuffle(index1)
        p1 = index1[:750]  # index for training set
        p2 = index1[751:]  # index for testing set
        dataset = np.hstack((data1, PN))  # dataset: including positive and negative samples
        train = dataset[p1]
        train = train.reshape(-1, 1)
        train_label = labels[p1]
        train_label = train_label.reshape(-1, 1)
        test = dataset[p2]
        test = test.reshape(-1, 1)
        test_label = labels[p2]
        test_label = test_label.reshape(-1, 1)
        clf = KNeighborsClassifier(n_neighbors=K)
        clf.fit(train, train_label)
        pre_label = clf.predict(test)  # prediction labels for testing data
        TP = 0
        TF=0
        # to calculate the probability of detection:
        for k in range(len(pre_label)):
            if pre_label[k] == test_label[k] == 1:
                TP = TP + 1
            if test_label[k]==0 and pre_label[k] == 1:
                TF  = TF +1
        Pd_k = TP / len(test_label[test_label == 1])
        Pf_k = TP / len(test_label[test_label == 0])
        PD_K.append(Pd_k)
        PF_K.append(Pf_k)
    PD_K = np.array(PD_K)
    PF_K= np.array(PF_K)
    # plt.plot(PD_K, 'k--o', label='KNN', linewidth=2), plt.legend(fontsize=18, loc='upper left')
    # plt.grid()
    # plt.show()
    return PD_K,PF_K


def HED_method(TH_SNR, PS, PN, K, TH):
    PD = []
    random.seed(113)
    SNR = np.linspace(0, 20, 21)
    m, n = PS.shape
    for i in range(len(SNR)):
        if i < TH_SNR:
            data1 = PS[i, :]
            label1 = np.ones(len(data1))  # label the positive samples with number 1
            label2 = np.zeros(len(PN))  # label the negative samples with number 1
            labels = np.hstack((label1, label2))  # all labels
            index1 = np.arange(0, len(labels), 1)  # index of labels and data
            random.shuffle(index1)  # shuffle the index of dataset
            p1 = index1[:750]  # 3/4 for the training set
            p2 = index1[751:]  # 1/4 for the testing set
            dataset = np.hstack((data1, PN))  # build the dataset by combine positive with negative samples
            train = dataset[p1]  # training data
            train = train.reshape(-1, 1)
            train_label = labels[p1]  # training labels
            train_label = train_label.reshape(-1, 1)
            test = dataset[p2]  # testing data
            test = test.reshape(-1, 1)
            test_label = labels[p2]  # test labels
            test_label = test_label.reshape(-1, 1)
            clf = KNeighborsClassifier(n_neighbors=K)  # build a KNN objection
            clf.fit(train, train_label)  # fitting the model
            pre_label = clf.predict(test)  # loading KNN model to classify
            TP = 0  # number of truly positive samples
            # to calculate the probability of detection:
            for k in range(len(pre_label)):
                if pre_label[k] == test_label[k] == 1:
                    TP = TP + 1
            Pd_k = TP / len(test_label[test_label == 1])
            PD.append(Pd_k)
        else:
            H0 = 0  # the number of positive samples
            H1 = 0  # the number of negative samples
            data = PS[i, :]  # loading data
            # compare to the threshold:
            for j in range(n):
                if data[j] > TH:  # busy
                    H1 = H1 + 1
                else:             # free
                    H0 = H0 + 1
            # probability of detection:
            Pd = H1 / n
            PD.append(Pd)
    PD = np.array(PD)
    return PD


if __name__ == '__main__':
    # loading data :
    PS = np.load("dataset_signals.npy")  # Power of signal (21,500)
    PN = np.load("dataset_noise.npy")  # variance of noise (500,)
    noise = PN
    SNR = []
    for i in range(21):
        signal = PS[i, :]
        snr=sum(signal) / sum(noise)
        snr_db = 10 * np.log10(snr)
        snr_db = np.floor(snr_db)
        SNR.append(snr_db)
    SNR = np.array(SNR)
    SNR = SNR.tolist()
    TH = np.mean(PN)  # threshold of ED
    PD_E = ED_method(PS, TH)  # receive the probability  detection  of ED method
    PD_K, PF_K = KNN_method(PS, PN, 3)  # receive the probability  detection  of KNN method
    TH_SNR = 5  # threshold of SNR
    K = 3  # nearest neighbors of KNN
    PD = HED_method(TH_SNR, PS, PN, K, TH)   # receive the probability  detection  of HED method
    # Result visualization:
    xi = np.linspace(0, 20, 21)
    plt.plot(xi,PD_E, 'r-*', label='ED'), plt.legend(fontsize=18)
    plt.plot(xi,PD, 'g--o', linewidth=4, label='HED'), plt.legend(fontsize=18)
    plt.plot(xi,PD_K, 'b--x', label='KNN', linewidth=2), plt.legend(fontsize=18, loc='lower right')
    plt.grid()
    # plt.axis([0, 20, 0, 1])
    plt.xticks(xi, SNR)
    plt.show()
    plt.plot(xi, PD_K)
    plt.plot(xi, PF_K)
    plt.show()

