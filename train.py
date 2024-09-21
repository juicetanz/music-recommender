import torch
from torch import nn
from torch.utils.data import DataLoader

import tqdm

device = 'cuda'
#if torch.cuda.is_available():
    #device = 'cuda'

BATCH_SIZE = 1
EPOCHS = 10

CHECKPOINT_PATH = 'checkpoint.pt'
MODEL_PATH = 'model_best.pt'
DATA_PATH = 'images.json'

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten(start_dim=0)
        self.linear1=nn.Linear(in_features=7543360,out_features=128)
        self.linear2=nn.Linear(in_features=128,out_features=1)
        self.output=nn.Sigmoid()
    
    def forward(self,input_data):
        x=self.conv1(input_data)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        logits=self.linear2(x)
        output=self.output(logits)

        return output

def train_single_epoch(model,dataloader,loss_fn,optimizer,device,model_best):
    for waveform,label in tqdm.tqdm(dataloader):
        waveform=waveform.to(device)
        # label=pt.from_numpy(numpy.array(label))
        label=label.to(device)
        # calculate loss and preds
        logits=model(waveform)
        loss=loss_fn(logits.float().unsqueeze(dim=1),label.float())
        # backpropogate the loss and update the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #save checkpoint
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f'Checkpoint saved at {CHECKPOINT_PATH} with loss: {loss.item()}')
    #save if best model
    if loss.item() < model_best:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'Best model saved at {MODEL_PATH} with loss improvement: {model_best - loss.item()}')
        return loss.item()
    else:
        return model_best

def train(model,dataloader,loss_fn,optimizer,device,epochs):
    model_best = 10
    for i in tqdm.tqdm(range(epochs)):
        print(f"Epoch: {i+1}")
        model_best = train_single_epoch(model,dataloader,loss_fn,optimizer,device,model_best)
        print('-------------------------------------------')
    print('Finished Training')

dataset = torch.load(DATA_PATH)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNN().cuda() #.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS)
