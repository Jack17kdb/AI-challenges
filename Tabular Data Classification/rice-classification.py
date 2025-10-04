# 0: Gonen, 1: Jasmine
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

df = pd.read_csv("rice-data.csv")
df.head()

df.dropna(inplace=True)
df.drop('id', axis=1, inplace=True)

original_df = df.copy()

for column in df.columns:
	df[column] = df[column]/df[column].abs().max()

df.head()

x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

class dataset(Dataset):
	def __init__(self, x, y):
		self.x = torch.tensor(x, dtype = torch.float32).to(device)
		self.y = torch.tensor(y, dtype = torch.float32).to(device)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		return self.x[index], self.y[index]

training_data = dataset(x_train, y_train)
validation_data = dataset(x_val, y_val)
testing_data = dataset(x_test, y_test)

train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=8, shuffle=True)

class myModel(nn.Module):
	def __init__(self):
		super(myModel, self).__init__()

		self.input_layer = nn.Linear(x.shape[1], 10)
		self.linear = nn.Linear(10, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.input_layer(x)
		x = self.linear(x)
		x = self.sigmoid(x)

		return x

model = myModel().to(device)

summary(model, (x.shape[1],))

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr = 1e-3)

total_loss_train_plot = []
total_loss_val_plot = []
total_acc_train_plot = []
total_acc_val_plot = []

epochs = 10
for epoch in range(epochs):
	total_loss_train = 0
	total_acc_train = 0
	total_loss_val = 0
	total_acc_val = 0

	for data in train_dataloader:
		inputs, labels = data
		prediction = model(inputs).squeeze(1)
		
		batch_loss = criterion(prediction, labels)
		total_loss_train += batch_loss.item()

		acc = ((prediction).round() == labels).sum().item()
		total_acc_train += acc

		batch_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

	with torch.no_grad():
		for data in val_dataloader:
			inputs, labels = data
			prediction = model(inputs).squeeze(1)

			batch_loss = criterion(prediction, labels)
			total_loss_val += batch_loss.item()

			acc = ((prediction).round() == labels).sum().item()
			total_acc_val += acc

	total_loss_train_plot.append(round(total_loss_train/1000 ,4))
	total_loss_val_plot.append(round(total_loss_val/1000, 4))

	total_acc_train_plot.append(round(total_acc_train/training_data.__len__() * 100, 4))
	total_acc_val_plot.append(round(total_acc_val/validation_data.__len__() * 100, 4))

	print(f'''Epoch No: {epoch + 1} \n Train loss: {round(total_loss_train/1000 ,4)} \n Train accuracy: {round(total_acc_train/training_data.__len__() * 100, 4)}
		      \n Validation loss: {round(total_loss_val/1000, 4)} \n Validation accuracy: {round(total_acc_val/validation_data.__len__() * 100, 4)} \n''')

	print("=" * 25)

with torch.no_grad():
	total_test_loss = 0
	total_test_acc = 0
	for data in test_dataloader:
		inputs, labels = data
		prediction = model(inputs).squeeze(1)

		batch_loss = criterion(prediction, labels).item()
		total_test_loss += batch_loss

		acc = ((prediction).round() == labels).sum().item()
		total_test_acc += acc

print(f"Test accuracy: {round(total_test_acc/testing_data.__len__() * 100, 4)}")

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

axs[0].plot(total_loss_train_plot, label = 'Training loss')
axs[0].plot(total_loss_val_plot, label = 'Validation loss')
axs[0].set_title('Training and Validation loss over epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0, 2])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label = 'Training accuracy')
axs[1].plot(total_acc_val_plot, label = 'Validation accuracy')
axs[1].set_title('Training and Validation accuracy over epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim([0, 100])
axs[1].legend()

plt.show()

area = 2383 / original_df['Area'].abs().max()
MajorAxisLength = 98 / original_df['MajorAxisLength'].abs().max()
MinorAxisLength = 78 / original_df['MinorAxisLength'].abs().max()
Eccentricity = 42 / original_df['Eccentricity'].abs().max()
ConvexArea = 22 / original_df['ConvexArea'].abs().max()
EquivDiameter = 38 / original_df['EquivDiameter'].abs().max()
Extent = 87 / original_df['Extent'].abs().max()
Perimeter = 933 / original_df['Perimeter'].abs().max()
Roundness = 732 / original_df['Roundness'].abs().max()
AspectRation = 31 / original_df['AspectRation'].abs().max()

my_pred = model(torch.tensor([area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation], dtype=torch.float32).to(device))
print(f"Prediction (0: Gonen, 1: Jasmine): {my_pred}")

