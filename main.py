import time

import torch
import torch.nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
from pickle import dump
from dataset import ImageDataset
from model import resnet_regression

#==================================side Image root=====================================
num_of_picture_train = 200
num_of_picture_test = 37

transforms_ = transforms.Compose([
    transforms.Resize((256,256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_base_before_root = '/home/ylab3/dataset_365/side_images/before/train/'
train_base_after_root = '/home/ylab3/dataset_365/side_images/after/train/'
test_base_before_root = '/home/ylab3/dataset_365/side_images/before/train/'
test_base_after_root = '/home/ylab3/dataset_365/side_images/after/train/'

train_before_root = [train_base_before_root+str(i+1)+'.jpg' for i in range(num_of_picture_train)]
train_after_root = [train_base_after_root+str(i+1)+'.jpg' for i in range(num_of_picture_train)]
test_before_root = [test_base_before_root+str(i+1)+'.jpg' for i in range(num_of_picture_test)]
test_after_root = [test_base_after_root+str(i+1)+'.jpg' for i in range(num_of_picture_test)]

train_all_picture_root = train_before_root+train_after_root
test_all_picture_root = test_before_root+test_after_root
#===============================side bmi root============================
train_diagnose_data = pd.read_excel('/home/ylab3/dataset_365/diagnose_BMI/train.xlsx')
test_diagnose_data = pd.read_excel('/home/ylab3/dataset_365/diagnose_BMI/test.xlsx')

train_before_data = []
train_after_data = []
test_before_data = []
test_after_data = []

train_diagnose_data = train_diagnose_data.drop(['Unnamed: 0'], axis=1)
test_diagnose_data = test_diagnose_data.drop(['Unnamed: 0'], axis=1)


for i in range(num_of_picture_train) :
    train_before_data.append(train_diagnose_data['Before_BMI'][i])
    train_after_data.append(train_diagnose_data['After_BMI'][i])
train_all_data = train_before_data + train_after_data

for i in range(num_of_picture_test) :
    test_before_data.append(test_diagnose_data['Before_BMI'][i])
    test_after_data.append(test_diagnose_data['After_BMI'][i])
test_all_data = test_before_data + test_after_data

train_all_data = np.array(train_all_data)
train_all_data = train_all_data.reshape(num_of_picture_train*2,1)
min_max_scaler = preprocessing.MinMaxScaler()
train_data_scaled = min_max_scaler.fit_transform(train_all_data)
dump(min_max_scaler, open('bmi_minmaxscaler.pkl','wb'))

test_all_data = np.array(test_all_data)
test_all_data = test_all_data.reshape(num_of_picture_test*2,1)
test_data_scaled = min_max_scaler.transform(test_all_data)
#============================load image======================================

train_dataset = ImageDataset(train_all_picture_root,train_data_scaled,transforms_)
test_dataset = ImageDataset(test_all_picture_root,test_data_scaled,transforms_)

train_dataloader = DataLoader(train_dataset, batch_size=200, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=200, num_workers=4)

#==================================model=====================================
model = resnet_regression()
model.cuda()

criterion_loss = torch.nn.MSELoss()
criterion_loss.cuda()

lr = 0.0002
n_epoch = 1000
check_term = 1

optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.5,0.999))

small_loss = torch.tensor(9999.999)
small_loss.cuda()

train_loss_list = []
test_loss_list = []


for epoch in range(n_epoch) :
    start_time = time.time()
    for i, batch in enumerate(train_dataloader):
        model.train()
        image = batch['image'].cuda()
        bmi = batch['bmi'].cuda()
        bmi = bmi.type(torch.cuda.FloatTensor)

        optimizer.zero_grad()
        prediction = model(image)
        train_loss = criterion_loss(prediction,bmi)

        train_loss.backward()
        optimizer.step()

    train_loss_list.append(train_loss.item())

    if epoch % check_term == 0 :
        model.eval()
        val_data = next(iter(test_dataloader))
        test_picture = val_data['image'].cuda()
        test_bmi = val_data['bmi'].cuda()
        test_bmi = test_bmi.type(torch.cuda.FloatTensor)

        prediction_test = model(test_picture)
        test_loss = criterion_loss(prediction_test, test_bmi)

        if test_loss.item()<small_loss.item() :
            test_loss_list.append(test_loss.item())
            torch.save(model, 'bmi_predictor_best.pt')

    print('epoch : '+str(epoch)+' [train_loss] : '+str(train_loss.item())+' [test_loss] : '+str(test_loss.item()))

x = np.arange(1,1001,1)

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(x,train_loss_list, label = 'train_loss', color = 'royalblue')
plt.legend(loc=0)
plt.title('train_loss',fontsize=20)
plt.subplot(2,1,2)
plt.plot(x,test_loss_list, label = 'test_loss', color = 'blue')
plt.legend(loc=0)
plt.title('test_loss',fontsize=20)

plt.savefig('loss_fig.jpg')