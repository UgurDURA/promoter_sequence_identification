import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
df=pd.read_csv("train_data.csv")
np_array_all=df.to_numpy()
np_array_2R = np_array_all[np_array_all[:,2] == "chr2R"]
# print(np_array_2R)
np_array_2R = np_array_2R[:,(0,1,7)]

# print(np_array_2R)
# print(np_array_2R.shape)
# print(len(np_array_2R))

np.random.shuffle(np_array_2R)
# print(np_array_2R)
# print(np_array_2R.shape)
# print(len(np_array_2R))

np_array_2R[:len(np_array_2R)//2,0] = 1 #validation
np_array_2R[len(np_array_2R)//2:,0] = 2 #test
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
categorical_vector_2R = np.zeros((len(np_array_2R),len(np_array_2R[0, 2])),dtype=np.int64)
categorical_vector_train = np.zeros((len(np_array_train),len(np_array_train[0, 2])),dtype=np.int64)

for i in range (0,len(np_array_train)):
    categorical_vector_train[i] = codetable[np.array(list(np_array_train[i,2])).view(np.int32)]
for i in range (0,len(np_array_2R)):
    categorical_vector_2R[i] = codetable[np.array(list(np_array_2R[i,2])).view(np.int32)]

# def create_ohe_tensor(array_data,categorical_vector):
#   #np_array_first_part=array_data[:,:2]
#   #np_array_first_part=(np_array_first_part).astype(np.int64)
#   #tensor_first_part=torch.tensor(np_array_first_part)
#   #tensor_second_part=torch.nn.functional.one_hot(torch.from_numpy(categorical_vector), num_classes=4)
#   #tensor_second_part=torch.flatten(tensor_second_part,1,2)
#   #tensor_data=torch.cat((tensor_first_part,tensor_second_part),1)
#   return tensor_data

#tensor_train=create_tensor(np_array_train,categorical_vector_train)
#tensor_2R=create_tensor(np_array_2R,categorical_vector_2R)
train_labels=torch.tensor(np_array_train[:,1].astype(np.int64))
valtest_labels=torch.tensor(np_array_2R[:,1].astype(np.int64))
train_samples = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector_train), num_classes=4)
valtest_samples = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector_2R), num_classes=4)
# print(train_labels.shape)
# print(valtest_labels.shape)
# print(train_samples.shape)
# print(valtest_samples.shape)
# print(train_labels.dtype)
# print(valtest_labels.dtype)
# print(train_samples.dtype)
# print(valtest_samples.dtype)
# print(valtest_samples[:len(np_array_2R)//2].shape)
# print(valtest_samples[len(np_array_2R)//2:].shape)
train_dataset=torch.utils.data.TensorDataset(train_samples,train_labels)
val_dataset=torch.utils.data.TensorDataset(valtest_samples[:len(np_array_2R)//2],valtest_labels[:len(np_array_2R)//2])
test_dataset=torch.utils.data.TensorDataset(valtest_samples[len(np_array_2R)//2:],valtest_labels[len(np_array_2R)//2:])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64) #have validation come at the same order every time -default shuffle = False
test_dataloader = DataLoader(test_dataset, batch_size=64) #have test come at the same order every time -default shuffle = False

# features,labels=next(iter(train_dataloader))
# print(features[0])
# print(labels[0])










