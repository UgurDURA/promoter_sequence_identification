import torch
import pandas as pd
import numpy as np
df=pd.read_csv("train_data.csv")
np_array_all=df.to_numpy()
np_array_2R = np_array_all[np_array_all[:,2] == "chr2R"]
print(np_array_2R)
np_array_2R = np_array_2R[:,(0,1,7)]

print(np_array_2R)
print(np_array_2R.shape)
print(len(np_array_2R))

np.random.shuffle(np_array_2R)
print(np_array_2R)
print(np_array_2R.shape)
print(len(np_array_2R))

np_array_2R[:len(np_array_2R)//2,0] = 1
np_array_2R[len(np_array_2R)//2:,0] = 2
np_array_train = np_array_all[np_array_all[:,2] != "chr2R"]
np_array_train = np_array_train[:,(0,1,7)]
np_array_train[:,0] = 0
# print(np_array_train)
# print(np_array_train.shape)

# First, we exploit the fact that numbers are also integers (ASCII code)
# To build a vector which maps letters to an index
codetable = np.zeros(256,np.int64)
for ix,nt in enumerate(["A","C","G","T"]):
  codetable[ord(nt)] = ix
# Now we use numpy indexing, using the letters in our sequence as index
# to extract the correct positions from our code table
categorical_vector_2R = np.zeros((len(np_array_2R),len(np_array_2R[0, 2])))
categorical_vector_train = np.zeros((len(np_array_train),len(np_array_train[0, 2])))

for i in range (0,len(np_array_train)):
    categorical_vector_train[i] = codetable[np.array(list(np_array_train[i,2])).view(np.int32)]

print(categorical_vector_train.shape)
print(categorical_vector_train)

for i in range (0,len(np_array_2R)):
    categorical_vector_2R[i] = codetable[np.array(list(np_array_2R[i,2])).view(np.int32)]

print(categorical_vector_2R.shape)
print(categorical_vector_2R)

#Based on first tutorial of ml4rg, check if we want it as numpy or tensor and act accordingly. 
# To convert this into a one-hot representation, we can use a pytorch function
#one_hot = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector), num_classes=4)
#After this step we have one hot encodings and we can add data to pytorch dataset


# count1=0
# count2=0
# for i in range (len(np_array_2R)):
#     if np_array_2R [i][0]==1:
#         count1+=1
#     else:
#         count2+=1

# print(count1)
# print(count2)


#train_tensor=torch.tensor(np_array)
#print(train_tensor,train_tensor.size())