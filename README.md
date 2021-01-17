# CANSOFCOM

## Problem
The Canadian Special Operations Forces Command wants to classify drones based on the RADAR signals that bounce off of them.
The drone actually distorts the returned frequency: the blades are moving fast, so some micro-doppler frequencies are returned.
Each drone has a unique return signal pattern, determined by its dimensions, number of rotors, radial speed of the rotors, number of blades per rotor, etc.

## Parameters
### Drone parameters
- $N$: The number of blades in a single rotor.
- $f_{rot}$: (1/s) the frequency of rotation of the rotor. This is drone-dependent, but could be \[50, 18\].
- $L_1$: (meters) The distance of the blade roots from the center of rotation of the rotor. For now, assumed to be 0.
- $L_2$: (meters) The distance of the blade tips from the center of rotation of the rotor. Differs depending on drone, generally 0.05 to 0.25.

### Scenario parameters:
- $A_r$: a scale factor, arbitrary constant. Set to 1 for now.
- $R$: (meters) The distance from the radar to the center of rotation (the rotor). Could be in the 1000 to 5000 range (a few kilometers). Assumed to be 0 for this hackathon.
- $\theta$: (radians) the positive angle between the plane of rotation of the rotor and the line of sight from the radar to the center of rotation (of the rotor). In the range \[0, $pi/2$\].

- $V_{rad}$: (rad/s) the radial velocity of the center of rotation (of the rotor) with respect to the RADAR. Assumed to be 0 for this hackathon, but this would affect doppler considerations.


### RADAR parameters:
- $f_c$: (1/s) the transmitted RADAR frequency. In the range \[10GHz, 94 GHz\].
- $\lambda$: (meters) the wavelength of the transmitted signal. lambda = c/f_c, where c is the speed of light in m/s.


- $t$: (s) the time, our real-valued continuous input.
- $f_s$: (1/s) the sampling frequency (how often we sample our RADAR return signal). 10 KHz for X-Band RADAR, 26KHz for W-Band RADAR.

---

## Overview
* `CANSOFCOM.ipynb` - annotated overview of the whole repository, as well as results.
* `signalgenerator.py` - the home for the function that builds $\psi$, and for generating data from sampling $\psi$.
* `classifier.py` - definitions of the Convolutional Classifier Neural Networks.
* `train.py` - training script for training a convnet on the generated data.
* `dataset_generation.py` - script that generates the filesystem of SW STFTs for all the different frequencies, SNRs, etc.
* `configurations.py` - place for storing dictionaries about scenarios and drones
* `helpers.py` - analysis tools
* `test.py` - big function for testing specific neural networks (TODO: make more general)
* `visualize.py` - example for plotting a fourier transform, time-domain signals, etc.
* `fourier.py` - provides a function to generate short and long-window STFTs.


## Dependencies



## History
This repository was built for the CANSOFCOM Drone Classification Machine Learning challenge of Hack The North 2020++.
You can see the original challenge in CANSOFCOM_Challenge.pdf.


