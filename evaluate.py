import pandas as pd
from cnn import CNN
import numpy as np

def evaluate(model):
    raw_df = pd.read_csv("E:\\Data\\kaggle-mnist\\test.csv")

    data = np.reshape(raw_df.values, (len(raw_df), 28, 28, 1))

    prob, pred = model.eval(data)
    result = pd.DataFrame({"Label":pred})
    print(result)

    print("DONE")
        