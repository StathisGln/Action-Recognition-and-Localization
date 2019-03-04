import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tcn_net import tcn_net
from simple_dataset import Video

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding


class DataParallelModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)

        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)

        self.block3 = nn.Linear(20, 20)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        t = torch.Tensor([1,2,2,3])
        z = torch.rand(3,16,112,112)
        return z

    def __len__(self):
        return self.len


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 1
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# generate model
classes = ['__background__', 'brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
           'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
           'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
           'swing_baseball', 'walk' ]

model = Model(input_size, output_size)
model = tcn_net(classes, 16, 112)
model.create_architecture()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)


model.to(device)
model.eval()

sample_size = 112
sample_duration = 16

dataset_folder = '/gpu-data2/sgal/JHMDB-act-detector-frames'
splt_txt_path  = '/gpu-data2/sgal/splits'
boxes_file     = '/gpu-data2/sgal/poses.json'
classes = ['__background__', 'brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


cls2idx = {classes[i]: i for i in range(0, len(classes))}


# # get mean
mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                             ToTensor(),
                             Normalize(mean, [1, 1, 1])])
temporal_transform = LoopPadding(sample_duration)

data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
             temporal_transform=temporal_transform, json_file = boxes_file,
             split_txt_path=splt_txt_path, mode='train', classes_idx=cls2idx)
data_loader = torch.utils.data.DataLoader(data, batch_size=2,
                                          shuffle=True)



rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
epochs = 20
for ep in range(epochs):
    for data in data_loader:
        clips,  (h, w), target, gt_tubes, n_frames = data
        vid = clips.to(device)
        n_f = n_frames.to(device)
        output = model(vid,None, None, n_f,max_dim=1)
        print("Outside: input size", input.size(),
              "output_size", output.size())
