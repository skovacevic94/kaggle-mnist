import pandas as pd
from cnn import CNN
import numpy as np

def evaluate(model):
    raw_df = pd.read_csv("E:\\Data\\kaggle-mnist\\test.csv")
    result = pd.DataFrame({"ImageId", "Label"})

    data = np.reshape(raw_df.values, (len(raw_df), 28, 28, 1))

    prob, pred = model.eval(data)


    print("DONE")
        