import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math
import os
import joblib
import pydot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
print(tf.__version__)

seq_length = 7 # Six past transactions followed by current transaction
df = pd.read_csv('test_220_100k.csv')
duplicate_rows = df[df.duplicated(['User', 'Card'])] # find duplicate rows

new_indices = []
count = 0
for index, row in df.iterrows():
    # find the first of each user card entry (from uniques)
    if index not in duplicate_rows.index:
    # check if the user/card for the last index are the same as the user/card for the first index
        num_iter = 0
        row_count = df.shape[0]
        first = df.loc[count + num_iter]
        while (True): 
            last_index = count + (seq_length - 1) + num_iter
            if row_count > last_index:
                last = df.loc[last_index]
            else: break
            if first['User'] == last['User'] and first['Card'] == last['Card']:
                first = df.loc[count + num_iter]
                # if there are exactly 7 consecutive entries (or more than 7 consecutive entries)
                if (last['Index'] - first['Index'] == seq_length - 1): 
                    for i in range(seq_length):
                        new_indices.append(count + i + num_iter) # append 7 consecutive entries
                    num_iter += seq_length        
                else: # if there are less than 7 consecutive entries, skip them
                    num_consecutive = 0
                    for i in range(seq_length): # count the number of consecutive entries (to know how many to skip)
                        curr_data = df.loc[count + num_iter + i]
                        next_data = df.loc[count + num_iter + i + 1]
                        if next_data['Index'] - curr_data['Index'] == 1:
                            num_consecutive += 1
                        else: break
                    num_iter += (num_consecutive + 1)
            else: break

    count += 1

# write to a new csv file
df.loc[new_indices].to_csv('new_indices.csv', index=False)
