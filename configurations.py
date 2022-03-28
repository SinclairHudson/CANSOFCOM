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
    "SNR": None,
    "deterministic": True
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
    "SNR": None,
    "deterministic": True
}

configforHERM = {
    "R": 0,
    "V_rad": 0,
    "A_r": 1,
    "N": 4,
    "L_1": 0.25,
    "L_2": 1,
    "f_rot": 40,
    "lamb": 0.02,
    "theta": 0,
    "f_c": c/0.02,  # must be c/lamb
    "SNR": None,
    "deterministic": True
}

config3 = {
    "R": 0,
    "V_rad": 0,
    "A_r": 1,
    "N": 2,
    "L_1": 0.25,
    "L_2": 1,
    "f_rot": 120,
    "lamb": 0.2,
    "theta": 0.1,
    "f_c": c/0.2,
    "SNR": None,
    "deterministic": True
}

scenarioXband= {
    "lamb": 0.02998,  # X band has about a 3 cm wavelength
    "f_c": c/0.02998,
    "SNR": 10
}

scenarioWband= {
    "lamb": 0.003189,  # W band has wavelength about 3 mm
    "f_c": c/0.003189,
    "SNR": None
}

djimavicair2 = {
    "name": "DJI_Mavic_Air_2",
    "N": 2,
    "L_1": 0.005,
    "L_2": 0.07,
    # https://mavicpilots.com/threads/mavic-average-rpm.4982/
    "f_rot": 91.66,
}

# taken from my own drone
djimavicmini = {
    "name": "DJI_Mavic_Mini",
    "N": 2,  # two blades per rotor
    "L_1": 0.005,  # 5 mm
    "L_2": 0.035,  # 3.5 cm
    # https://forum.dji.com/thread-214023-1-1.html
    "f_rot": 160,  # 9600 rpm
}

djimatrice300rtk = {
    # https://www.bhphotovideo.com/c/product/1565975-REG/dji_cp_en_00000270_01_matrice_300_series_propeller.html
    "name": "DJI_Matrice_300_RTK",
    "N": 2,  # two blades per rotor
    "L_1": 0.05,
    "L_2": 0.2665,
    "f_rot": 70,  # a guess
}

parrotdisco = {
    # https://www.amazon.com/Parrot-PF070252-Genuine-Disco-Propeller/dp/B01MSMUWW4
    "name": "Parrot_Disco",
    "N": 2,  # two blades per rotor
    "L_1": 0.01,
    "L_2": 0.104,
    "f_rot": 40,  # a guess (lower because rotor isn't used for lift)
}

djiphantom4 = {
    # https://store.dji.com/ca/product/phantom-4-series-low-noise-propellers
    "name": "DJI_Phantom_4",
    "N": 2,  # two blades per rotor
    "L_1": 0.006,
    "L_2": 0.05,
    # https://phantompilots.com/threads/motor-rpm.16886/
    "f_rot": 116,
}
