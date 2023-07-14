import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
df=pd.read_csv("cofactor_expression_data.csv")
np_array_all=df.to_numpy()
np_array_2R = np_array_all[np_array_all[:,2] == "chr2R"]
# print(np_array_2R)
data_range=(7,9,10,18,19,22)
np_array_2R = np_array_2R[:,data_range]

#print(np_array_2R)
# print(np_array_2R.shape)
# print(len(np_array_2R))

np.random.shuffle(np_array_2R)
# print(np_array_2R)
# print(np_array_2R.shape)
# print(len(np_array_2R))
np_array_train = np_array_all[np_array_all[:,2] != "chr2R"]
np_array_train = np_array_train[:,data_range]
# print(np_array_train)
# print(np_array_train.shape)

# First, we exploit the fact that numbers are also integers (ASCII code)
# To build a vector which maps letters to an index
codetable = np.zeros(256,np.int64)
for ix,nt in enumerate(["A","C","G","T"]):
  codetable[ord(nt)] = ix
# Now we use numpy indexing, using the letters in our sequence as index
# to extract the correct positions from our code table
# print(np_array_2R)
# print(len(np_array_2R))
categorical_vector_2R = np.zeros((len(np_array_2R),len(np_array_2R[0, 0])),dtype=np.int64)
categorical_vector_train = np.zeros((len(np_array_train),len(np_array_train[0, 0])),dtype=np.int64)
#print(categorical_vector_2R.shape)
for i in range (0,len(np_array_train)):
    categorical_vector_train[i] = codetable[np.array(list(np_array_train[i,0])).view(np.int32)]
for i in range (0,len(np_array_2R)):
    categorical_vector_2R[i] = codetable[np.array(list(np_array_2R[i,0])).view(np.int32)]

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(np_array_train)
train_labels=torch.tensor(np_array_train[:,1:].astype(np.float64))
valtest_labels=torch.tensor(np_array_2R[:,1:].astype(np.float64))
train_samples = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector_train), num_classes=4)
valtest_samples = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector_2R), num_classes=4)

train_labels = train_labels.to(device)
valtest_labels = valtest_labels.to(device)
train_samples = train_samples.to(device)
valtest_samples = valtest_samples.to(device)
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

# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=64) #have validation come at the same order every time -default shuffle = False
# test_dataloader = DataLoader(test_dataset, batch_size=64) #have test come at the same order every time -default shuffle = False

# features,labels=next(iter(train_dataloader))
# print(features[0])
# print(labels[0])

hparams =             {'batch_size_train': 128,#64, # number of examples per batch
                      'batch_size_vt':128,
                      'epochs': 60, # number of epochs SHOULD BE 100
                      #'early_stop': 10, # patience of 10 epochs to reduce training time; you can increase the patience to see if the model improves after more epochs
                      'lr': 0.001, # learning rate
                      #'n_conv_layer': 3, # number of convolutional layers
                      'num_filters1': 128, # number of filters/kernels in the first conv layer
                      'num_filters2': 60, # number of filters/kernels in the second conv layer
                      'num_filters3': 60, # number of filters/kernels in the third conv layer
                      'num_filters4': 120,
                      'kernel_size1': 7, # size of the filters in the first conv layer
                      'kernel_size2': 3, # size of the filters in the second conv layer
                      'kernel_size3': 5, # size of the filters in the third conv layer
                      'kernel_size4': 3,
                      'n_dense_layer': 1, # number of dense/fully connected layers
                      'dense_neurons1': 256, # number of neurons in the dense layer
                      'dense_neurons2': 256,
                      'dropout_prob': 0.3, # dropout probability
                      }
train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size_train'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=hparams['batch_size_vt']) #have validation come at the same order every time -default shuffle = False
test_dataloader = DataLoader(test_dataset, batch_size=hparams['batch_size_vt']) #have test come at the same order every time -default shuffle = False

#check initialization of layers (He, xavier. tensorflow probably initializes with Xavier) for convolutional layers
#can also check bias term to True to match tensorflow implementation
class DeepSTARR(nn.Module):
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        """
        super().__init__()
        self.hparams = hparams
      
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=self.hparams['num_filters1'], kernel_size=self.hparams['kernel_size1'], stride=1, bias=False, padding=(self.hparams['kernel_size1']-1)//2),
            nn.BatchNorm1d(self.hparams['num_filters1']),
            nn.ReLU(),
            nn.MaxPool1d(2,stride =2),
            #nn.Dropout(p = self.hparams['dropout_prob']),
            
            nn.Conv1d(in_channels=self.hparams['num_filters1'], out_channels=self.hparams['num_filters2'], kernel_size=self.hparams['kernel_size2'], stride=1, bias=False, padding=(self.hparams['kernel_size2']-1)//2),
            nn.BatchNorm1d(self.hparams['num_filters2']),
            nn.ReLU(),
            nn.MaxPool1d(2, stride = 2),
            nn.Dropout(p= self.hparams['dropout_prob']),

            nn.Conv1d(in_channels=self.hparams['num_filters2'], out_channels=self.hparams['num_filters3'], kernel_size=self.hparams['kernel_size3'], stride=1, bias=False, padding=(self.hparams['kernel_size3']-1)//2),
            nn.BatchNorm1d(self.hparams['num_filters3']),
            nn.ReLU(),
            nn.MaxPool1d(2, stride = 2),
            nn.Dropout(p= self.hparams['dropout_prob']),

            nn.Conv1d(in_channels=self.hparams['num_filters3'], out_channels=self.hparams['num_filters4'], kernel_size=self.hparams['kernel_size4'], stride=1, bias=False, padding=(self.hparams['kernel_size4']-1)//2),
            nn.BatchNorm1d(self.hparams['num_filters4']),
            nn.ReLU(),
            nn.MaxPool1d(2, stride = 2),
            nn.Dropout(p= self.hparams['dropout_prob']),
        )

        self.classifier =  nn.Sequential(
            nn.Linear(960, self.hparams['dense_neurons2']),
            nn.BatchNorm1d(self.hparams['dense_neurons2']),
            nn.ReLU(),
            nn.Dropout(p= self.hparams['dropout_prob']),

            nn.Linear(self.hparams['dense_neurons2'],self.hparams['dense_neurons1']),
            nn.BatchNorm1d(self.hparams['dense_neurons1']),
            nn.ReLU(),
            nn.Dropout(p= self.hparams['dropout_prob']),

            #nn.Linear(self.hparams['dense_neurons1'],5)
            #nn.Sigmoid()
        )
        self.linear_p65=nn.Linear(self.hparams['dense_neurons1'],1)
        self.linear_p300=nn.Linear(self.hparams['dense_neurons1'],1)
        self.linear_gfzf=nn.Linear(self.hparams['dense_neurons1'],1)
        self.linear_chro=nn.Linear(self.hparams['dense_neurons1'],1)
        self.linear_mof=nn.Linear(self.hparams['dense_neurons1'],1)


        pass

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = x.to(torch.float32)
        #to avoid RuntimeError: Input type (torch.cuda.LongTensor) and weight type (torch.cuda.FloatTensor) should be the same
        #could converting from floattensor (int64?) to float32 cause a problem with values precision?
        x = self.model(x)
        #print(x.shape)
        #x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1) #keep nsamples dim and flatten the 2 rest dimensions to pass in linear
        #print(x.shape)
        #print(x.shape)
        x = self.classifier(x)
        p65=self.linear_p65(x)
        p300=self.linear_p300(x)
        gfzf=self.linear_gfzf(x)
        chro=self.linear_chro(x)
        mof=self.linear_mof(x)

        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return torch.cat((p65,p300,gfzf,chro,mof),dim=1)


#Also include test, right now only doing train and validation !!!!!!!!!!!!!! TRAIN FOR MORE EPOCHS, MAKE ARCHITECTURE BIGGGER, HYPERPARAMETERS CHECK
#Taken and changed by i2dl TUM course exercises
def train_model(model, train_loader, val_loader,test_loader):
    """
    Train the model for a number of epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=model.hparams['lr'])
    # size_train = len(train_loader.dataset)
    # size_val = len(val_loader.dataset)
    size_tloader = len(train_loader)
    size_vloader = len(val_loader)
    # max_t_acc = 0
    # max_v_acc = 0 
    for epoch in range(model.hparams['epochs']):
        
        training_loss = 0
        # Training stage, where we want to update the parameters.
        model.train()  # Set the model to training mode
        #correct_samples=0
        for i, (batch_samples, batch_labels) in enumerate (train_loader):
            optimizer.zero_grad()
            #samples, labels = batch["image"].to(device), batch["keypoints"].to(device)
            pred = model(batch_samples)
            # print(pred.shape)
            # print(pred.dtype)
            # print(batch_labels.shape)
            # print(batch_labels.dtype)
            # print(pred)
            # print(batch_labels)
            # print(pred.shape)
            # print(batch_labels.shape)
            #print(batch_labels.unsqueeze(1).float())
            #print(pred)
            # print((pred >= 0.5).float())
            mseloss = nn.MSELoss()
            batch_labels = batch_labels.to(torch.float32)   #need to convert in order to calculate loss as a float and not a double
            loss=mseloss(pred,batch_labels) #unsqueeze and float to match dimensions and dtype
            loss.backward()  # Stage 2: Backward().
            optimizer.step() # Stage 3: Update the parameters.
             
            # binary_pred = (pred >= 0.5).float()   #>= so that we can study more sequences as positive
            
            training_loss += loss.item()
            # correct_samples += binary_pred.eq(batch_labels.unsqueeze(1).float()).sum().item()

        print("Epoch", epoch+1,":")
        print("Average training loss is: {:.3f}".format(training_loss/size_tloader))
        # print("Training accuracy is: {:.3f}".format(correct_samples/size_train *100), "%")
        # max_t_acc = max(max_t_acc,correct_samples/size_train *100)
        # Validation stage, where we don't want to update the parameters. Pay attention to the model.eval() line
        # and "with torch.no_grad()" wrapper.
        model.eval()
        # correct_samples = 0
        validation_loss = 0
        with torch.no_grad():
            for i, (batch_samples, batch_labels) in enumerate(val_loader):
                pred = model(batch_samples)
                mseloss = nn.MSELoss()
                batch_labels = batch_labels.to(torch.float32)   #need to convert in order to calculate loss as a float and not a double
                loss = mseloss(pred,batch_labels)
                # binary_pred = (pred >= 0.5).float()

                validation_loss += loss.item()
                # correct_samples += binary_pred.eq(batch_labels.unsqueeze(1).float()).sum().item()
        print("Epoch", epoch+1,":")
        print("Average validation loss is: {:.3f}".format(validation_loss/size_vloader))
        # print("Validation accuracy is: {:.3f}".format(correct_samples/size_val *100), "%")
        # max_v_acc = max(max_v_acc,correct_samples/size_val *100)
        # print("\nBest training accuracy: {:.3f}".format(max_t_acc), "%")
        # print("Best validation accuracy: {:.3f}".format(max_v_acc), "%\n")
    
    #do this to save best model every time
    # torch.save(model.state_dict(), "trained_model.pth")
    # print("Model saved successfully.")

    # loaded_model = DeepSTARR(hparams)
    # loaded_model.load_state_dict(torch.load("trained_model.pth"))
    # loaded_model.to(device)
    
    test_loss = 0
    size_testloader = len(test_loader)
    with torch.no_grad():
        for i, (batch_samples, batch_labels) in enumerate(test_loader):
            pred = model(batch_samples)
            mseloss = nn.MSELoss()
            batch_labels = batch_labels.to(torch.float32)   #need to convert in order to calculate loss as a float and not a double
            loss = mseloss(pred,batch_labels)
            test_loss += loss.item()
    print("Average test loss is: {:.3f}".format(test_loss/size_testloader))
    
    #PCC calculation for validation set
    cofactor_names = ["p65", "Nej_p300", "gfzf", "Chro", "Mof"]
    input = valtest_samples[:len(np_array_2R)//2]
    output = model(input)
    pred = output.detach().cpu().numpy()
    target = valtest_labels[:len(np_array_2R)//2].detach().cpu().numpy()
    PCC = []
    for i in range(len(cofactor_names)):
        PCC.append(np.corrcoef(pred[:, i], target[:, i])[1][0].round(3))
    df = pd.DataFrame(PCC, cofactor_names, columns=["PCC"])
    print(df)
    def scatter_density(i):

        g = sns.jointplot(x=target[:, i], y=pred[:, i], kind="kde", fill=True, color="red")
        g.ax_marg_x.remove()
        g.ax_marg_y.remove()

        # Regression line

        x0, x1 = g.ax_joint.get_xlim()
        y0, y1 = g.ax_joint.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        g.ax_joint.plot(lims, lims, 'w', linestyle='dashed', transform=g.ax_joint.transData, color='grey')

        g.ax_joint.set_aspect('equal')

        plt.xlabel('Real Value')
        plt.ylabel('Predicted Value')
        plt.title(str(cofactor_names[i] + '(PCC = ' + str(PCC[i]) + ')'))

        plt.show()
    for i in range(0,len(target[0])):
        scatter_density(i)

model = DeepSTARR(hparams)
train_model(model.to(device), train_dataloader, val_dataloader,test_dataloader)



