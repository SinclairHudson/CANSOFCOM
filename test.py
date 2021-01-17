import torch.nn as nn
import time
import wandb
import torch
import torchvision
import torchvision.datasets as ds
import numpy as np
from classifier import RadarDroneClassifierW, RadarDroneClassifierX
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc, roc_auc_score
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import torch.nn as nn
from helpers import confuse, to_one_hot_vector

c = {
    "epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 64,
    "SNR": 5,
    "f_s": 26000,
}


def dataloader(file_extension):
    data = np.load(file_extension)
    return data


def testclassifier(f_s, SNR):

    num_classes = 5
    c = {
        "epochs": 20,
        "learning_rate": 0.001,
        "batch_size": 64,
        "SNR": SNR,
        "f_s": f_s,
    }

    softmax = nn.Softmax(dim=1)  # class dimension, not batch
    testds = ds.DatasetFolder(
        f"testset/{c['f_s']}fs/{c['SNR']}SNR", dataloader, extensions=("npy",))
    testLoader = torch.utils.data.DataLoader(
        testds, c["batch_size"], shuffle=True)

    class_map = ["DJI_Matrice_300_RTK", "DJI_Mavic_Air_2",
                 "DJI_Mavic_Mini", "DJI_Phantom_4", "Parrot_Disco"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if c["f_s"] == 26_000:
        net = RadarDroneClassifierW().to(device)
    else:
        net = RadarDroneClassifierX().to(device)

    net.load_state_dict(torch.load(
        f"e{c['epochs']}SNR{c['SNR']}f_s{c['f_s']}.pt"))
    net.eval()

    confm = np.zeros((5, 5), dtype=int)
    correct = 0
    total = 0

    predicted_big = np.zeros((0, 5))
    true_labels = np.zeros((0, 5))
    for i, data in enumerate(testLoader):
        inputs, y = data
        inputs = inputs.to(device)
        y = y.to(device)

        yhat = net(inputs.float())

        # search for max along dimension 1, also note that index.
        softout = softmax(yhat)

        onehottrue = to_one_hot_vector(num_classes, y.cpu().numpy())
        softout = softout.cpu()

        predicted_big = np.concatenate(
            (softout.numpy(), predicted_big), axis=0)
        true_labels = np.concatenate((onehottrue, true_labels), axis=0)

        confidence, predicted = torch.max(softout.data, 1)

        l, p = torch.Tensor.cpu(y).numpy(), torch.Tensor.cpu(predicted).numpy()

        cm = confuse(l, p, 5)

        confm = np.add(confm, cm)
        correct += (predicted == y.cpu()).sum().item()
        total += c["batch_size"]

    fig, ax = plot_confusion_matrix(conf_mat=confm,
                                    show_normed=True,
                                    colorbar=True,
                                    class_names=class_map)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(true_labels[:, i],
                                                            predicted_big[:, i])
        average_precision[i] = average_precision_score(
            true_labels[:, i], predicted_big[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(true_labels.ravel(),
                                                                    predicted_big.ravel())

    average_precision["micro"] = average_precision_score(true_labels, predicted_big,
                                                         average="micro")

    print('Average precision score, micro-averaged over all classes: {0:0.4f}'
          .format(average_precision["micro"]))

    print("accuracy: ", correct/total)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predicted_big[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), predicted_big.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i],
                 lw=2, label=f"{class_map[i]} ROC curve (area = {roc_auc[i]:02f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curves')
    plt.legend(loc="lower right")

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))

    # plt.figure()
    # plt.title("Recall, class-wise")
    # plt.xlabel('Examples, ordered by confidence')
    # plt.ylabel('Recall')
    # for c in range(num_classes):
        # line = plt.plot(recall[c])

    # plt.legend(lines, labels)


    plt.show()


with torch.no_grad():
    testclassifier(26_000, 5)
