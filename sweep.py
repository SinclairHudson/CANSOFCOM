from train import OTFTrain
from test import testclassifier
from drone_constants import class_map
import numpy as np
import matplotlib.pyplot as plt



def sweep_f_cs():

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

    f_cs = [9.0e10, 7.0e10, 5.0e10, 3.0e10, 1.0e10]

    for f_c in f_cs:
            print(f"starting training of f_c = {f_c}")
            conf["f_c"] = f_c
            OTFTrain(conf)

    APs = [[] for x in range(len(class_map))]
    micro_averages = []
    for f_c in f_cs:
        print(f"starting testing of f_c = {f_c}")
        conf["f_c"] = f_c
        results = testclassifier(f"models/{str(conf)}.pt", dataset_size=10_000, 
                       sample_length=conf["signal_duration"], f_s=conf["f_s"], SNR=conf["SNR"])
        micro_averages.append(results["AP score"]['micro'])
        for x in range(len(class_map)):
            APs[x].append(results["AP score"][x])

    for x in range(len(class_map)):
        plt.plot(f_cs, APs[x], label=class_map[x])

    plt.plot(f_cs, micro_averages, label="average")

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
    for f_s in f_ss:
            print(f"starting training of f_s = {f_s}")
            conf["f_s"] = f_s
            OTFTrain(conf)

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

    for x in range(len(class_map)):
        plt.plot(f_ss, APs[x], label=class_map[x])

    plt.plot(f_ss, micro_averages, label="average")
    plt.xlabel('Sampling Frequency')
    plt.ylabel('Average Precision')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.show()

sweep_f_ss()
