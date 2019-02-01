import numpy as np

def _get_next_batch(dataframe, batch_size):
        batch_dataframe = dataframe.sample(batch_size)

        data = np.reshape(batch_dataframe.drop(['label'], axis=1).values, (batch_size, 28, 28, 1))
        labels = np.reshape(batch_dataframe[['label']].values, (batch_size))
        
        return data, labels
