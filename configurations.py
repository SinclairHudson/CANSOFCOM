# This file holds all the configuration dictionaries that I wil use for simulation

c = 2.998e8  # speed of light in m/s

paperconfig1 = {
    "R": 0,
    "V_rad": 0,
    "A_r": 1,
    "N": 4,
    "L_1": 0.25,
    "L_2": 1,
    "f_rot": 40,
    "lamb": 0.2,
    "theta": 0,
    "f_c": c/0.2,
    "SNR": None
}


paperconfig2 = {
    "R": 0,
    "V_rad": 0,
    "A_r": 1,
    "N": 5,
    "L_1": 0.25,
    "L_2": 1,
    "f_rot": 40,
    "lamb": 0.2,
    "theta": 0,
    "f_c": c/0.2,  # must be c/lamb
    "SNR": None
}

config3 = {
    "R": 0,
    "V_rad": 0,
    "A_r": 1,
    "N": 4,
    "L_1": 0.25,
    "L_2": 1,
    "f_rot": 120,
    "lamb": 0.2,
    "theta": 0.1,
    "f_c": c/0.2,
    "SNR": None
}

scenario = {
    "R": 0,
    "V_rad": 0,
    "A_r": 1,
    "lamb": 0.2,
    "theta": 0.1,
    "f_c": c/0.2,
    "SNR": 0
}

djimavicair2 = {
    "N": 2,
    "L_1": 0.25,
    "L_2": 1,
    "f_rot": 120,
}

# taken from my own drone
djimavicmini = {
    "N": 2,  # two blades per rotor
    "L_1": 0.005,  # 5 mm
    "L_2": 0.035,  # 3.5 cm
    "f_rot": 120,  # a guess
}
