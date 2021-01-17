import numpy as np
from sklearn.metrics import confusion_matrix

def confuse(l, p, num_classes):
    cm = confusion_matrix(l, p)
    classesrep = cm.shape[0]
    if classesrep < num_classes:
        pad = num_classes - classesrep
        cm = np.pad(cm, ((0, pad), (0, pad)), mode='constant', constant_values=(0,0))  # zero pad

    assert cm.shape == (num_classes, num_classes)
    return cm
