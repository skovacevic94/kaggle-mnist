import pandas as pd
from sklearn.model_selection import train_test_split
from cnn import CNN
from evaluate import evaluate
import matplotlib.pyplot as plt
import numpy as np

batch_size = 128
dropout = 0.15
random_seed = 2

raw_df = pd.read_csv("E:\\Data\\kaggle-mnist\\train.csv")
train_df, validation_df = train_test_split(raw_df, test_size=0.2, random_state=random_seed)

model = CNN('C:/Users/v-stkova/source/repos/kaggle-mnist/tmp/model.ckpt')
model.train(train_df, validation_df, epoch=2000, batch_size=batch_size, learning_rate=0.001, dropout=dropout)
