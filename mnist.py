import pandas as pd
from sklearn.model_selection import train_test_split
from cnn import CNN
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32

raw_df = pd.read_csv("E:\\Data\\kaggle-mnist\\train.csv")
train_df, validation_df = train_test_split(raw_df, test_size=0.2)

model = CNN()
model.train(train_df, validation_df, 100, 64, 0.0001)

random_image = validation_df.sample(n=1)
label = np.reshape(random_image[['label']].values, (1))
print(label)
data = np.reshape(random_image.drop(['label'], axis=1).values, newshape=(28, 28))
plt.imshow(data)
plt.show()
prob, pred = model.eval(data)
print("End")