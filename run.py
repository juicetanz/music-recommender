import torch
import torchaudio
from torch import nn

import sys
import os

MODEL_PATH = 'data/model_best.pt'

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

model = CNN()
model.load_state_dict(torch.load(MODEL_PATH))

def predict(model, input):
    min = 1885824
    wf, sr = torchaudio.load(input)
    wf = wf[:,:min]
    wf = torch.mean(wf, dim=0)
    wf = torch.unsqueeze(wf, 0)
    wf = torch.unsqueeze(wf, 0)
    
    with torch.no_grad():
        model.eval()
        
        return model(wf)
user_favorite = predict(model, sys.argv[1])

def predict_candidates(model, items):
    values = []
    for idx, item in enumerate(items):
        values.append(predict(model, sys.argv[2] + item))
        print(f'Songs loaded: {str(idx + 1)} / {str(len(items))}')
    return values

candidates = os.listdir(sys.argv[2])[:10]
candidate_values = predict_candidates(model, candidates)

closest_index = min(range(len(candidate_values)), key=lambda i: abs(candidate_values[i] - user_favorite))
recommendation = candidates[closest_index]
print(recommendation[:-4])
