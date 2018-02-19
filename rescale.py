import pandas as pd
import numpy as np
from scipy.special import expit, logit

def rescale(x):
    return expit(logit(x) - 0.5)

if __name__ == "__main__":
    sub_num = "32"
    sub = pd.read_csv("submissions\\submission" + sub_num + ".csv")
    labels = [c for c in sub.columns][2:]
    for label in labels:
        sub[label] = sub[label].apply(rescale)
    sub.to_csv("submissions\\submission" + sub_num + "rescaled.csv", index=False, encoding="utf-8")