from train import OTFTrain, train
from test import testclassifier
from drone_constants import class_map
import numpy as np
import matplotlib.pyplot as plt


def sweep_f_cs():

    conf = {
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 128,
        "SNR": 10,
        "f_s": 26_000,
        "f_c": 9.4e10,
        "signal_duration": 0.15,
        "train_set_size": 10_000,
        "test_set_size": 2_000,
    }

    f_cs = [9.0e10, 7.0e10, 5.0e10, 3.0e10, 1.0e10]

    # for f_c in f_cs:
            # print(f"starting training of f_c = {f_c}")
            # conf["f_c"] = f_c
            # OTFTrain(conf)

    APs = [[] for x in range(len(class_map))]
    micro_averages = []
    accuracies = []
    for f_c in f_cs:
        print(f"starting testing of f_c = {f_c}")
        conf["f_c"] = f_c
        results = testclassifier(f"models/{str(conf)}.pt", conf=conf, dataset_size=10_000)
        micro_averages.append(results["AP score"]['micro'])
        accuracies.append(results["accuracy"])
        for x in range(len(class_map)):
            APs[x].append(results["AP score"][x])

    for x in range(len(class_map)):
        plt.plot(f_cs, APs[x], label=class_map[x])

    plt.plot(f_cs, micro_averages, label="average")
    plt.plot(f_cs, accuracies, label="accuracy")

    plt.xlabel('carrier frequency')
    plt.ylabel('Average Precision')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.show()


def sweep_f_ss():

    conf = {
        "epochs": 75,
        "learning_rate": 0.001,
        "batch_size": 128,
        "SNR": 10,
        "f_s": 26_000,
        "f_c": 9.4e10,
        "signal_duration": 0.15,
        "train_set_size": 10_000,
        "test_set_size": 2_000,
    }

    f_ss = [26_000, 21_000, 16_000, 11_000, 6_000, 1_000]

    # training the models
    # for f_s in f_ss:
            # print(f"starting training of f_s = {f_s}")
            # conf["f_s"] = f_s
            # OTFTrain(conf)

    # testing the models
    APs = [[] for x in range(len(class_map))]
    micro_averages = []
    accuracies = []
    for f_s in f_ss:
        print(f"starting testing of f_s = {f_s}")
        conf["f_s"] = f_s
        results = testclassifier(f"models/{str(conf)}.pt", dataset_size=10_000,
                       sample_length=conf["signal_duration"], f_s=conf["f_s"], SNR=conf["SNR"])
        micro_averages.append(results["AP score"]['micro'])
        accuracies.append(results["accuracy"])
        for x in range(len(class_map)):
            APs[x].append(results["AP score"][x])

    np.save("sweeps/f_sAccuracy.npy", np.array(accuracies))
    np.save("sweeps/f_sAPs.npy", np.array(APs))
    np.save("sweeps/f_smicro_averages.npy", np.array(micro_averages))

    plt.figure()
    plt.plot(f_ss, accuracies, label="accuracy")
    for x in range(len(class_map)):
        plt.plot(f_ss, APs[x], label=class_map[x])

    plt.plot(f_ss, micro_averages, label="average")
    plt.xlabel('Sampling Frequency')
    plt.ylabel('Average Precision')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.show()

def OfflineMetrics():
    accuracies = np.load("sweeps/f_sAccuracy.npy")
    APs = np.load("sweeps/f_sAPs.npy")
    micro_averages = np.load("sweeps/f_smicro_averages.npy")

    f_ss = [26_000, 21_000, 16_000, 11_000, 6_000, 1_000]

    for x in range(len(class_map)):
        plt.plot(f_ss, APs[x], label=class_map[x])

    plt.plot(f_ss, micro_averages, label="average")
    plt.plot(f_ss, accuracies, label="accuracy")
    plt.xlabel('Sampling Frequency')
    plt.ylabel('Average Precision')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.show()


def sweep_SNR():
    print("running an SNR sweep")
    conf = {
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 128,
        "SNR": 10,
        "f_s": 26_000,
        "f_c": 9.4e10,
        "signal_duration": 0.15,
        "train_set_size": 10_000,
        "test_set_size": 2_000,
    }

    SNR = [15, 12.5, 10, 7.5, 5, 2.5]

    for S in SNR:
            conf["SNR"] = S
            OTFTrain(conf, overwrite=True)

    APs = [[] for x in range(len(class_map))]
    micro_averages = []
    accuracies = []
    for S in SNR:
        conf["SNR"] = S
        results = testclassifier(f"models/{str(conf)}.pt", conf=conf, dataset_size=10_000)
        micro_averages.append(results["AP score"]['micro'])
        accuracies.append(results["accuracy"])
        for x in range(len(class_map)):
            APs[x].append(results["AP score"][x])

    np.save("sweeps/SNRAccuracy.npy", np.array(accuracies))
    np.save("sweeps/SNRAPs.npy", np.array(APs))
    np.save("sweeps/SNRmicro_averages.npy", np.array(micro_averages))

    plt.figure()
    for x in range(len(class_map)):
        plt.plot(SNR, APs[x], label=class_map[x])

    plt.plot(SNR, micro_averages, label="average")
    plt.plot(SNR, accuracies, label="accuracy")

    plt.xlabel('SNR')
    plt.ylabel('Average Precision')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    OfflineMetrics()
