from train import OTFTrain

conf = {
    "epochs": 1,
    "learning_rate": 0.001,
    "batch_size": 128,
    "SNR": 10,
    "f_s": 26_000,
    "f_c": 9.4e10,
    "signal_duration": 0.15,
    "train_set_size": 10_000,
    "test_set_size": 2_000,
}

# 94 GHz to 10 Ghz, or from around W band to X band
f_cs = [9.0e10, 7.0e10, 5.0e10, 3.0e10, 1.0e10]

for f_c in f_cs:
    print(f"starting training of f_c = {f_c}")
    conf["f_c"] = f_c
    OTFTrain(conf)
