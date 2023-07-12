import torch
import torch.nn as nn
import time
from tqdm import tqdm

from src.Network.network_helpers import plot_curves

class cofactor_DeepSTARR(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=self.hparams['num_filters1'], kernel_size=self.hparams['kernel_size1'], stride=1, bias=False, padding=(self.hparams['kernel_size1']-1)//2),
            nn.BatchNorm1d(self.hparams['num_filters1']),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(in_channels=self.hparams['num_filters1'], out_channels=self.hparams['num_filters2'], kernel_size=self.hparams['kernel_size2'], stride=1, bias=False, padding=(self.hparams['kernel_size2']-1)//2),
            nn.BatchNorm1d(self.hparams['num_filters2']),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(p=self.hparams['dropout_prob']),
            nn.Conv1d(in_channels=self.hparams['num_filters2'], out_channels=self.hparams['num_filters3'], kernel_size=self.hparams['kernel_size3'], stride=1, bias=False, padding=(self.hparams['kernel_size3']-1)//2),
            nn.BatchNorm1d(self.hparams['num_filters3']),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(p=self.hparams['dropout_prob']),
            nn.Conv1d(in_channels=self.hparams['num_filters3'], out_channels=self.hparams['num_filters4'], kernel_size=self.hparams['kernel_size4'], stride=1, bias=False, padding=(self.hparams['kernel_size4']-1)//2),
            nn.BatchNorm1d(self.hparams['num_filters4']),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(p=self.hparams['dropout_prob']),
        )

        self.classifier = nn.Sequential(
            nn.Linear(960, self.hparams['dense_neurons2']),
            nn.BatchNorm1d(self.hparams['dense_neurons2']),
            nn.ReLU(),
            nn.Dropout(p=self.hparams['dropout_prob']),
            nn.Linear(self.hparams['dense_neurons2'], self.hparams['dense_neurons1']),
            nn.BatchNorm1d(self.hparams['dense_neurons1']),
            nn.ReLU(),
            nn.Dropout(p=self.hparams['dropout_prob']),
            nn.Linear(self.hparams['dense_neurons1'], 15)
        )

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = x.to(torch.float32)  
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

def cofactor_train_model(model, train_loader, val_loader, test_loader):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model.hparams['lr'])

    size_train = len(train_loader.dataset)
    size_val = len(val_loader.dataset)
    size_tloader = len(train_loader)
    size_vloader = len(val_loader)
    max_t_acc = 0
    max_v_acc = 0 

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(model.hparams['epochs']):
        training_loss = 0
        model.train()

        correct_samples = 0

        progress_bar = tqdm(enumerate(train_loader), total=size_tloader, leave=True, desc='Epoch {}/{}'.format(epoch+1, model.hparams['epochs']))

        for i, (batch_samples, batch_labels) in progress_bar:
            optimizer.zero_grad()
            pred = model(batch_samples)
            mseloss = nn.MSELoss()
            batch_labels = batch_labels.to(torch.float32)
            loss = mseloss(pred, batch_labels)
            loss.backward()
            optimizer.step()

            binary_pred = (pred >= 0.5).float()
            training_loss += loss.item()
            correct_samples += torch.round(binary_pred).eq(batch_labels.unsqueeze(1).float()).sum().item()

            progress_bar.set_postfix({'Training Loss': training_loss / (i+1)})

        epoch_train_loss = training_loss / size_tloader
        epoch_train_acc = correct_samples / size_train * 100
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)

        model.eval()
        correct_samples = 0
        validation_loss = 0

        with torch.no_grad():
            for i, (batch_samples, batch_labels) in enumerate(val_loader):
                pred = model(batch_samples)
                mseloss = nn.MSELoss()
                batch_labels = batch_labels.to(torch.float32)
                loss = mseloss(pred, batch_labels)
                binary_pred = (pred >= 0.5).float()

                validation_loss += loss.item()
                correct_samples += torch.round(binary_pred).eq(batch_labels.unsqueeze(1).float()).sum().item()

        epoch_val_loss = validation_loss / size_vloader
        epoch_val_acc = correct_samples / size_val * 100
        val_loss_list.append(epoch_val_loss)
        val_acc_list.append(epoch_val_acc)

        print("Epoch", epoch+1, ":")
        print("Average training loss is: {:.3f}".format(epoch_train_loss))
        print("Training accuracy is: {:.3f}".format(epoch_train_acc), "%")
        print("Average validation loss is: {:.3f}".format(epoch_val_loss))
        print("Validation accuracy is: {:.3f}".format(epoch_val_acc), "%")

        max_t_acc = max(max_t_acc, epoch_train_acc)
        max_v_acc = max(max_v_acc, epoch_val_acc)
        print("\nBest training accuracy: {:.3f}".format(max_t_acc), "%")
        print("Best validation accuracy: {:.3f}".format(max_v_acc), "%\n")

    test_loss = 0
    size_testloader = len(test_loader)
    with torch.no_grad():
        for i, (batch_samples, batch_labels) in enumerate(test_loader):
            pred = model(batch_samples)
            mseloss = nn.MSELoss()
            batch_labels = batch_labels.to(torch.float32)
            loss = mseloss(pred, batch_labels)
            test_loss += loss.item()

    print("Average test loss is: {:.3f}".format(test_loss / size_testloader))