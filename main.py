import torch as tor
from torch import nn
from torch.nn import functional as fct
import os
import cv2
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.datasets as ds
from PIL import Image
import torchvision.transforms.functional as TF
if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')
import PIL

img_t = transforms.Compose([transforms.Resize(49),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
b_size = 600


dts =ds.CIFAR10(download=True,transform=img_t,root='./data')
dload=DataLoader(dts,batch_size=b_size,shuffle=True)

def to_img(x):
	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	x = x.view(x.size(0), 3, 49, 49)
	return x

def to_pimg(x):
	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	x = x.view(49, 49, 3)
	return x

class auto_enc(nn.Module):
	def __init__(self,in_ch):
		super().__init__()
		self.enc1=nn.Conv2d(in_channels=in_ch,out_channels=in_ch*16,kernel_size=5,stride=2)
		self.ebn1=nn.BatchNorm2d(16*in_ch)
#		self.enmp1=nn.MaxPool2d(2,stride=2)
		self.enc2=nn.Conv2d(in_channels=16*in_ch,out_channels=8*in_ch,kernel_size=5,stride=2)
#		self.enmp2=nn.MaxPool2d(2,stride=2)
		self.ebn2=nn.BatchNorm2d(8*in_ch)
		self.decup=nn.UpsamplingBilinear2d(scale_factor=2.4)
#		self.dec1=nn.ConvTranspose2d(8*in_ch,out_channels=16*in_ch,kernel_size=3,stride=2)
#		self.dec2=nn.ConvTranspose2d(16*in_ch,8*in_ch,kernel_size=5,stride=2)
		self.dconv5=nn.Conv2d(8*in_ch,16*in_ch,kernel_size=5,stride=2)

		self.dec3=nn.ConvTranspose2d(16*in_ch,8*in_ch,kernel_size=5,stride=2)
		self.dbn1=nn.BatchNorm2d(8*in_ch)
		self.dec4=nn.ConvTranspose2d(8*in_ch,in_ch,kernel_size=5,stride=2)


	def encode(self,x):
		x=fct.relu(self.enc1(x),True)
		x=self.ebn1(x)
#		x=self.enmp1(x)
		x=fct.relu(self.enc2(x),True)
		x=self.ebn2(x)
#		x=self.enmp2(x)
		return x

	def decode(self,x):
		x=self.decup(x)
	#	x=fct.relu(self.dec1(x),True)
	#	x=fct.relu(self.dec2(x),True)

		x=fct.relu(self.dconv5(x),True)
		x=fct.relu(self.dec3(x),True)
		x=self.dbn1(x)
		x=fct.tanh(self.dec4(x))
		return x

	def forward(self, x):
		x=self.encode(x)
		x=self.decode(x)
		return x


y= auto_enc(3).cuda()
a = tor.rand(1,3,49,49).cuda()
print(y.encode(a).shape)
print(y.decode(y.encode(a)).shape)

if os.path.exists("./conv_auto2.pth"):
	y.load_state_dict(tor.load("conv_auto30.pth"))
	y.eval()
	print('yes')


num_epochs = 0
learning_rate = 1e-3

criterion = nn.MSELoss()
optimizer = tor.optim.Adam(y.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in tqdm(range(0,num_epochs)):
	for d in tqdm(dload):
		img,_=d
		img= Variable(img).cuda()
		out=y(img)
		loss=criterion(out,img)
		y.zero_grad()
		loss.backward()
		optimizer.step()
	print(loss)
	if epoch%10==0:
		pic = to_img(out.cpu().data)
		pic2= to_img(img.cpu().data)
		save_image(pic, './dc_img/image_{}.png'.format(epoch))
		save_image(pic2,'./dc_img/image_o{}.png'.format(epoch))
		tor.save(y.state_dict(),'./conv_auto{}.pth'.format(epoch))







image=Image.open('img.jpg').resize((49,49),Image.BILINEAR)
x = TF.to_tensor(image)
plt.imshow(image)
plt.show()
#x.unsqueeze_(0)


