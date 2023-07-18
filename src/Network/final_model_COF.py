from turtle import ScrolledCanvas
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
# import shap
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
from parse_motifs import parse_motifs

df=pd.read_csv("cofactor_expression_data.csv")
np_array_all=df.to_numpy()
np_array_2R = np_array_all[np_array_all[:,2] == "chr2R"]

data_range=(7,9,10,18,19,22)
np_array_2R = np_array_2R[:,data_range]


#np.random.shuffle(np_array_2R)                    #NO SHUFFLING IN ORDER TO KNOW WHICH SEQUENCE EXACTLY DO I HAVE, IT MAKES IT EASIER TO INTERPRET AND WE DON'T NEED SHUFFLE


np_array_train = np_array_all[np_array_all[:,2] != "chr2R"]
np_array_train = np_array_train[:,data_range]


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


train_dataset=torch.utils.data.TensorDataset(train_samples,train_labels)
val_dataset=torch.utils.data.TensorDataset(valtest_samples[:len(np_array_2R)//2],valtest_labels[:len(np_array_2R)//2])
test_dataset=torch.utils.data.TensorDataset(valtest_samples[len(np_array_2R)//2:],valtest_labels[len(np_array_2R)//2:])



hparams =             {'batch_size_train': 128,#64, # number of examples per batch
                      'batch_size_vt':128,
                      'epochs': 1, # number of epochs SHOULD BE 100
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
        
        x = torch.flatten(x, start_dim=1) #keep nsamples dim and flatten the 2 rest dimensions to pass in linear
        
        x = self.classifier(x)
        p65=self.linear_p65(x)
        p300=self.linear_p300(x)
        gfzf=self.linear_gfzf(x)
        chro=self.linear_chro(x)
        mof=self.linear_mof(x)

        return torch.cat((p65,p300,gfzf,chro,mof),dim=1)



#Taken and changed by i2dl TUM course exercises
def train_model(model, train_loader, val_loader,test_loader):
    """
    Train the model for a number of epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=model.hparams['lr'])
    
    size_tloader = len(train_loader)
    size_vloader = len(val_loader)
    
    min_validation_loss = 1000
    for epoch in range(model.hparams['epochs']):
        
        training_loss = 0
        # Training stage, where we want to update the parameters.
        model.train()  # Set the model to training mode
        #correct_samples=0
        for i, (batch_samples, batch_labels) in enumerate (train_loader):
            optimizer.zero_grad()
            #samples, labels = batch["image"].to(device), batch["keypoints"].to(device)
            pred = model(batch_samples)
            
            mseloss = nn.MSELoss()
            batch_labels = batch_labels.to(torch.float32)   #need to convert in order to calculate loss as a float and not a double
            loss=mseloss(pred,batch_labels) #unsqueeze and float to match dimensions and dtype
            loss.backward()  # Stage 2: Backward().
            optimizer.step() # Stage 3: Update the parameters.
             
            
            
            training_loss += loss.item()
            

        print("Epoch", epoch+1,":")
        print("Average training loss is: {:.3f}".format(training_loss/size_tloader))
        
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
                
        print("Epoch", epoch+1,":")
        print("Average validation loss is: {:.3f}".format(validation_loss/size_vloader))
        # if min_validation_loss > validation_loss/size_vloader:            #uncomment this to save best model best on val loss
        #     min_validation_loss = validation_loss/size_vloader
        #     torch.save(model.state_dict(), "COF5_model.pth")
        #     print("Model saved successfully.")

    model = DeepSTARR(hparams)
    model.load_state_dict(torch.load("COF5_model.pth"))
    model.to(device)
    model.eval()

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
    
    #PCC calculation for test set
    cofactor_names = ["p65", "Nej_p300", "gfzf", "Chro", "Mof"]
    input = valtest_samples[len(np_array_2R)//2:]
    output = model(input)
    pred = output.detach().cpu().numpy()
    target = valtest_labels[len(np_array_2R)//2:].detach().cpu().numpy()
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

   
    motifs=["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"]
    
    csv_files_p65=[]
    for i in range(0,len(motifs)):
        csv_files_p65.append(motifs[i]+"_p65.csv")

    csv_files_p300=[]
    for i in range(0,len(motifs)):
        csv_files_p300.append(motifs[i]+"_p300.csv")

    csv_files_gfzf=[]
    for i in range(0,len(motifs)):
        csv_files_gfzf.append(motifs[i]+"_gfzf.csv")

    csv_files_chro=[]
    for i in range(0,len(motifs)):
        csv_files_chro.append(motifs[i]+"_chro.csv")

    csv_files_mof=[]
    for i in range(0,len(motifs)):
        csv_files_mof.append(motifs[i]+"_mof.csv")
    cofs=["p65","p300","gfzf","chro","mof"]
    cof_txt_files=[]
    for i in range (0,len(cofs)):
        cof_txt_files.append("coordinates_"+cofs[i]+".txt")
    parse_inputs=[]
    for i in range(0,len(csv_files_p65)):
        parse_inputs.append((csv_files_p65[i],cof_txt_files[0]))
    for i in range(0,len(csv_files_p300)):
            parse_inputs.append((csv_files_p300[i],cof_txt_files[1]))
    for i in range(0,len(csv_files_gfzf)):
            parse_inputs.append((csv_files_gfzf[i],cof_txt_files[2]))
    for i in range(0,len(csv_files_chro)):
            parse_inputs.append((csv_files_chro[i],cof_txt_files[3]))
    for i in range(0,len(csv_files_mof)):
            parse_inputs.append((csv_files_mof[i],cof_txt_files[4]))
    

    p_65_dicts=[]
    for i in range(0,19):
        p_65_dicts.append(parse_motifs(parse_inputs[i][0],parse_inputs[i][1]))
    p65_seqs_set=set()
    for i in range(0,len(p_65_dicts)):
         p65_seqs_set= p65_seqs_set.union(p_65_dicts[i].keys())
    #print((p65_seqs_set))
    print(len(p65_seqs_set))
    np_array_p65=np.array(list(p65_seqs_set))

    p_300_dicts=[]
    for i in range(19,38):
        p_300_dicts.append(parse_motifs(parse_inputs[i][0],parse_inputs[i][1]))       
    p300_seqs_set=set()
    for i in range(0,len(p_300_dicts)):
         p300_seqs_set= p300_seqs_set.union(p_300_dicts[i].keys())
    #print((p300_seqs_set))
    print(len(p300_seqs_set))
    np_array_p300=np.array(list(p300_seqs_set))

    gfzf_dicts=[]
    for i in range(38,57):
        gfzf_dicts.append(parse_motifs(parse_inputs[i][0],parse_inputs[i][1]))
    gfzf_seqs_set=set()
    for i in range(0,len(gfzf_dicts)):
         gfzf_seqs_set= gfzf_seqs_set.union(gfzf_dicts[i].keys())
    #print((gfzf_seqs_set))
    print(len(gfzf_seqs_set))
    np_array_gfzf=np.array(list(gfzf_seqs_set))

    chro_dicts=[]
    for i in range(57,76):
        chro_dicts.append(parse_motifs(parse_inputs[i][0],parse_inputs[i][1]))
    chro_seqs_set=set()
    for i in range(0,len(chro_dicts)):
         chro_seqs_set= chro_seqs_set.union(chro_dicts[i].keys())
    #print((chro_seqs_set))
    print(len(chro_seqs_set))
    np_array_chro=np.array(list(chro_seqs_set))

    mof_dicts=[]
    for i in range(76,95):
        mof_dicts.append(parse_motifs(parse_inputs[i][0],parse_inputs[i][1]))
    mof_seqs_set=set()
    for i in range(0,len(mof_dicts)):
         mof_seqs_set= mof_seqs_set.union(mof_dicts[i].keys())
    #print((mof_seqs_set))
    print(len(mof_seqs_set))
    np_array_mof=np.array(list(mof_seqs_set))

    codetable = np.zeros(256,np.int64)
    for ix,nt in enumerate(["A","C","G","T"]):
        codetable[ord(nt)] = ix
    
    
    categorical_vector_p65 = np.zeros((len(np_array_p65),len(np_array_p65[0])),dtype=np.int64)
    # #print(categorical_vector_2R.shape)
    for i in range (0,len(np_array_p65)):
         categorical_vector_p65[i] = codetable[np.array(list(np_array_p65[i])).view(np.int32)]
    
    cat_vector_p65 = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector_p65), num_classes=4)


    categorical_vector_p300 = np.zeros((len(np_array_p300),len(np_array_p300[0])),dtype=np.int64)
    # #print(categorical_vector_2R.shape)
    for i in range (0,len(np_array_p300)):
         categorical_vector_p300[i] = codetable[np.array(list(np_array_p300[i])).view(np.int32)]
    
    cat_vector_p300 = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector_p300), num_classes=4)

    categorical_vector_gfzf = np.zeros((len(np_array_gfzf),len(np_array_gfzf[0])),dtype=np.int64)
    # #print(categorical_vector_2R.shape)
    for i in range (0,len(np_array_gfzf)):
         categorical_vector_gfzf[i] = codetable[np.array(list(np_array_gfzf[i])).view(np.int32)]
    
    cat_vector_gfzf = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector_gfzf), num_classes=4)

    categorical_vector_chro = np.zeros((len(np_array_chro),len(np_array_chro[0])),dtype=np.int64)
    # #print(categorical_vector_2R.shape)
    for i in range (0,len(np_array_chro)):
         categorical_vector_chro[i] = codetable[np.array(list(np_array_chro[i])).view(np.int32)]
    
    cat_vector_chro = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector_chro), num_classes=4)

    categorical_vector_mof = np.zeros((len(np_array_mof),len(np_array_mof[0])),dtype=np.int64)
    # #print(categorical_vector_2R.shape)
    for i in range (0,len(np_array_mof)):
         categorical_vector_mof[i] = codetable[np.array(list(np_array_mof[i])).view(np.int32)]
    
    cat_vector_mof = torch.nn.functional.one_hot(torch.from_numpy(categorical_vector_mof), num_classes=4)

    cat_vector_p65 = cat_vector_p65.to(device)
    cat_vector_p300 = cat_vector_p300.to(device)
    cat_vector_gfzf = cat_vector_gfzf.to(device)
    cat_vector_chro = cat_vector_chro.to(device)
    cat_vector_mof = cat_vector_mof.to(device)

    torch.manual_seed(123)  
    np.random.seed(123)
    

    baseline_p65 = torch.zeros_like(cat_vector_p65)
    baseline_p300 = torch.zeros_like(cat_vector_p300)
    baseline_gfzf = torch.zeros_like(cat_vector_gfzf)
    baseline_chro = torch.zeros_like(cat_vector_chro)
    baseline_mof = torch.zeros_like(cat_vector_mof)
    
    ig = IntegratedGradients(model)

    batch_size=128

    if cat_vector_p65.shape[0] % batch_size != 0:
        idx = cat_vector_p65.shape[0]//batch_size + 1
        last_batch = cat_vector_p65.shape[0] % batch_size
    else:
        idx = cat_vector_p65.shape[0]//batch_size

    att_p65 = torch.tensor([])
    att_p65 = att_p65.to(device)
    for i in range (0,idx):
        if i != idx-1:
            attributions_p65, delta_p65 = ig.attribute(cat_vector_p65[i*128:i*128+128], baseline_p65[i*128:i*128+128], target=0, return_convergence_delta=True)
        else:
            attributions_p65, delta_p65 = ig.attribute(cat_vector_p65[i*128:i*128+last_batch], baseline_p65[i*128:i*128+last_batch], target=0, return_convergence_delta=True)
        torch.cuda.empty_cache()
        att_p65 = torch.cat((att_p65,attributions_p65), dim = 0)   


    if cat_vector_p300.shape[0] % batch_size != 0:
        idx = cat_vector_p300.shape[0]//batch_size + 1
        last_batch = cat_vector_p300.shape[0] % batch_size
    else:
        idx = cat_vector_p300.shape[0]//batch_size

    att_p300 = torch.tensor([])
    att_p300 = att_p300.to(device)
    for i in range (0,idx):
        if i != idx-1:
            attributions_p300, delta_p300 = ig.attribute(cat_vector_p300[i*128:i*128+128], baseline_p300[i*128:i*128+128], target=1, return_convergence_delta=True)
        else:
            attributions_p300, delta_p300 = ig.attribute(cat_vector_p300[i*128:i*128+last_batch], baseline_p300[i*128:i*128+last_batch], target=1, return_convergence_delta=True)
        torch.cuda.empty_cache()
        att_p300 = torch.cat((att_p300,attributions_p300), dim = 0) 

    if cat_vector_gfzf.shape[0] % batch_size != 0:
        idx = cat_vector_gfzf.shape[0]//batch_size + 1
        last_batch = cat_vector_gfzf.shape[0] % batch_size
    else:
        idx = cat_vector_gfzf.shape[0]//batch_size

    att_gfzf = torch.tensor([])
    att_gfzf = att_gfzf.to(device)
    for i in range (0,idx):
        if i != idx-1:
            attributions_gfzf, delta_gfzf = ig.attribute(cat_vector_gfzf[i*128:i*128+128], baseline_gfzf[i*128:i*128+128], target=2, return_convergence_delta=True)
        else:
            attributions_gfzf, delta_gfzf = ig.attribute(cat_vector_gfzf[i*128:i*128+last_batch], baseline_gfzf[i*128:i*128+last_batch], target=2, return_convergence_delta=True)
        torch.cuda.empty_cache()
        att_gfzf = torch.cat((att_gfzf,attributions_gfzf), dim = 0) 


    if cat_vector_chro.shape[0] % batch_size != 0:
        idx = cat_vector_chro.shape[0]//batch_size + 1
        last_batch = cat_vector_chro.shape[0] % batch_size
    else:
        idx = cat_vector_chro.shape[0]//batch_size

    att_chro = torch.tensor([])
    att_chro = att_chro.to(device)
    for i in range (0,idx):
        if i != idx-1:
            attributions_chro, delta_chro = ig.attribute(cat_vector_chro[i*128:i*128+128], baseline_chro[i*128:i*128+128], target=3, return_convergence_delta=True)
        else:
            attributions_chro, delta_chro = ig.attribute(cat_vector_chro[i*128:i*128+last_batch], baseline_chro[i*128:i*128+last_batch], target=3, return_convergence_delta=True)
        torch.cuda.empty_cache()
        att_chro = torch.cat((att_chro,attributions_chro), dim = 0)           #UNCOMMENT HERE AND UP

    if cat_vector_mof.shape[0] % batch_size != 0:
        idx = cat_vector_mof.shape[0]//batch_size + 1
        last_batch = cat_vector_mof.shape[0] % batch_size
    else:
        idx = cat_vector_mof.shape[0]//batch_size   

    att_mof = torch.tensor([])
    att_mof = att_mof.to(device)
    for i in range (0,idx):
        if i != idx-1:
            attributions_mof, delta_mof = ig.attribute(cat_vector_mof[i*128:i*128+128], baseline_mof[i*128:i*128+128], target=4, return_convergence_delta=True)
        else:
            attributions_mof, delta_mof = ig.attribute(cat_vector_mof[i*128:i*128+last_batch], baseline_mof[i*128:i*128+last_batch], target=4, return_convergence_delta=True)
        torch.cuda.empty_cache()
        att_mof = torch.cat((att_mof,attributions_mof), dim = 0)

    cof_motif_contr_dict={}
    cof_motif_contr_dict[cofactor_names[0]]=[]
    

    for i in range(0,len(p_65_dicts)):
        motif_array=[]
        for j in p_65_dicts[i].keys():
            average_motifs=[]
            #print(j)
            index_seq=list(p65_seqs_set).index(j)
            #print(index_seq)
            motif_occurences=p_65_dicts[i][j]
            #print(motif_occurences)
            
            for k in range(len(motif_occurences)):
                motif_start,motif_end=motif_occurences[k]
                average_motif=[]
                for l in range(motif_start,motif_end+1):
                    #print(att_p65[index_seq][l])
                    for nuc in att_p65[index_seq][l]:
                        if(nuc!=0):
                            average_motif.append(nuc)
                average_motif=sum(average_motif)/len(average_motif)
                average_motifs.append(average_motif)
            motif_array.append(average_motifs)    
        cof_motif_contr_dict[cofs[0]].append(motif_array)
    p65_arr_new=[]      
    for motif_arrays in cof_motif_contr_dict["p65"]:
        averages=[]
        for seq_averages in motif_arrays:
            for motif_average in seq_averages:
                averages.append(motif_average)
        p65_arr_new.append(averages)
    
    data_list=[]
    for tensors in p65_arr_new:
        data=[]
        for tensor in tensors:
            #print(tensor)
            data.append(tensor.item())
        data=np.array(data)
        data_list.append(data)
    
    fig, ax = plt.subplots()
    fig.set_figwidth(10,True)
    fig.set_figheight(3,True)
    # Creating plot
    #bp = ax.boxplot(data_list)
    ax.boxplot(data_list)
    ax.set_ylabel('P65 Contribution Scores')
    #ax.set_xticklabels(["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"])

    ax.set_xticklabels(["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"])
    ax.tick_params(axis='x', rotation=270)
    # show plot
    plt.show()

    
    cof_motif_contr_dict={}
    cof_motif_contr_dict[cofs[1]]=[]

    for i in range(0,len(p_300_dicts)):
        motif_array=[]
        for j in p_300_dicts[i].keys():
            average_motifs=[]
            #print(j)
            index_seq=list(p300_seqs_set).index(j)
            #print(index_seq)
            motif_occurences=p_300_dicts[i][j]
            #print(motif_occurences)
            
            for k in range(len(motif_occurences)):
                motif_start,motif_end=motif_occurences[k]
                average_motif=[]
                for l in range(motif_start,motif_end+1):
                    #print(att_p65[index_seq][l])
                    for nuc in att_p300[index_seq][l]:
                        if(nuc!=0):
                            average_motif.append(nuc)
                average_motif=sum(average_motif)/len(average_motif)
                average_motifs.append(average_motif)
            motif_array.append(average_motifs)    
        cof_motif_contr_dict[cofs[1]].append(motif_array)
    p300_arr_new=[]      
    for motif_arrays in cof_motif_contr_dict["p300"]:
        averages=[]
        for seq_averages in motif_arrays:
            for motif_average in seq_averages:
                averages.append(motif_average)
        p300_arr_new.append(averages)
    
    data_list=[]
    for tensors in p300_arr_new:
        data=[]
        for tensor in tensors:
            #print(tensor)
            data.append(tensor.item())
        data=np.array(data)
        data_list.append(data)
    
    fig, ax = plt.subplots()
    fig.set_figwidth(10,True)
    fig.set_figheight(3,True)
    # Creating plot
    #bp = ax.boxplot(data_list)
    ax.boxplot(data_list)
    ax.set_ylabel('P300 Contribution Scores')
    #ax.set_xticklabels(["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"])

    ax.set_xticklabels(["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"])
    ax.tick_params(axis='x', rotation=270)
    # show plot
    plt.show()


    cof_motif_contr_dict={}
    cof_motif_contr_dict[cofs[2]]=[]

    for i in range(0,len(gfzf_dicts)):
        motif_array=[]
        for j in gfzf_dicts[i].keys():
            average_motifs=[]
            #print(j)
            index_seq=list(gfzf_seqs_set).index(j)
            #print(index_seq)
            motif_occurences=gfzf_dicts[i][j]
            #print(motif_occurences)
            
            for k in range(len(motif_occurences)):
                motif_start,motif_end=motif_occurences[k]
                average_motif=[]
                for l in range(motif_start,motif_end+1):
                    #print(att_p65[index_seq][l])
                    for nuc in att_gfzf[index_seq][l]:
                        if(nuc!=0):
                            average_motif.append(nuc)
                average_motif=sum(average_motif)/len(average_motif)
                average_motifs.append(average_motif)
            motif_array.append(average_motifs)    
        cof_motif_contr_dict[cofs[2]].append(motif_array)
    gfzf_arr_new=[]      
    for motif_arrays in cof_motif_contr_dict["gfzf"]:
        averages=[]
        for seq_averages in motif_arrays:
            for motif_average in seq_averages:
                averages.append(motif_average)
        gfzf_arr_new.append(averages)
    
    data_list=[]
    for tensors in gfzf_arr_new:
        data=[]
        for tensor in tensors:
            #print(tensor)
            data.append(tensor.item())
        data=np.array(data)
        data_list.append(data)
    
    fig, ax = plt.subplots()
    fig.set_figwidth(10,True)
    fig.set_figheight(3,True)
    ax.boxplot(data_list)
    ax.set_ylabel('Gfzf Contribution Scores')
    #ax.set_xticklabels(["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"])

    ax.set_xticklabels(["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"])# Creating axes instance
     #mb need to change axis , also give names of MOTIF and COF to axis
 
    # Creating plot
    # show plot
    plt.show()


    cof_motif_contr_dict={}
    cof_motif_contr_dict[cofs[3]]=[]

    for i in range(0,len(chro_dicts)):
        motif_array=[]
        for j in chro_dicts[i].keys():
            average_motifs=[]
            #print(j)
            index_seq=list(chro_seqs_set).index(j)
            #print(index_seq)
            motif_occurences=chro_dicts[i][j]
            #print(motif_occurences)
            
            for k in range(len(motif_occurences)):
                motif_start,motif_end=motif_occurences[k]
                average_motif=[]
                for l in range(motif_start,motif_end+1):
                    #print(att_p65[index_seq][l])
                    for nuc in att_chro[index_seq][l]:
                        if(nuc!=0):
                            average_motif.append(nuc)
                average_motif=sum(average_motif)/len(average_motif)
                average_motifs.append(average_motif)
            motif_array.append(average_motifs)    
        cof_motif_contr_dict[cofs[3]].append(motif_array)
    chro_arr_new=[]      
    for motif_arrays in cof_motif_contr_dict["chro"]:
        averages=[]
        for seq_averages in motif_arrays:
            for motif_average in seq_averages:
                averages.append(motif_average)
        chro_arr_new.append(averages)
    
    data_list=[]
    for tensors in chro_arr_new:
        data=[]
        for tensor in tensors:
            #print(tensor)
            data.append(tensor.item())
        data=np.array(data)
        data_list.append(data)
    
 
    fig, ax = plt.subplots()
    fig.set_figwidth(10,True)
    fig.set_figheight(3,True)
    ax.boxplot(data_list)
    ax.set_ylabel('Chro Contribution Scores')
    #ax.set_xticklabels(["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"])

    ax.set_xticklabels(["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"])# Creating axes instance
     #mb need to change axis , also give names of MOTIF and COF to axis
    plt.show()


    cof_motif_contr_dict={}
    cof_motif_contr_dict[cofs[4]]=[]

    for i in range(0,len(mof_dicts)):
        motif_array=[]
        for j in mof_dicts[i].keys():
            average_motifs=[]
            
            index_seq=list(mof_seqs_set).index(j)
            
            motif_occurences=mof_dicts[i][j]
            
            
            for k in range(len(motif_occurences)):
                motif_start,motif_end=motif_occurences[k]
                average_motif=[]
                for l in range(motif_start,motif_end+1):
                    #print(att_p65[index_seq][l])
                    for nuc in att_mof[index_seq][l]:
                        if(nuc!=0):
                            average_motif.append(nuc)
                average_motif=sum(average_motif)/len(average_motif)
                average_motifs.append(average_motif)
            motif_array.append(average_motifs)    
        cof_motif_contr_dict[cofs[4]].append(motif_array)
    mof_arr_new=[]      
    for motif_arrays in cof_motif_contr_dict["mof"]:
        averages=[]
        for seq_averages in motif_arrays:
            for motif_average in seq_averages:
                averages.append(motif_average)
        mof_arr_new.append(averages)
    
    data_list=[]
    for tensors in mof_arr_new:
        data=[]
        for tensor in tensors:
            #print(tensor)
            data.append(tensor.item())
        data=np.array(data)
        data_list.append(data)

    fig, ax = plt.subplots()
    fig.set_figwidth(10,True)
    fig.set_figheight(3,True)
    ax.boxplot(data_list)
    ax.set_ylabel('Mof Contribution Scores')

    ax.set_xticklabels(["Ohler6","Ebox","DCE1","TATAbox", "Ohler7", "DPE_extended", "DCE2", "BREu","DRE", "DCE3","Ohler8","INR","Ohler1","BREd","DPE_Kadonaga","MTE","TCT","DPE","TC_17_Zabidi"])# Creating axes instance
    plt.show()

model = DeepSTARR(hparams)
train_model(model.to(device), train_dataloader, val_dataloader,test_dataloader)



