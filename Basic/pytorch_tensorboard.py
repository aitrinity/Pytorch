import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import sys
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

# create a summary writer
writer = SummaryWriter("runs/mnist")


# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper Parameters
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2 
batch_size = 64
learning_rate = 0.001

#Mnist Dataset

#Dowloading train and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform= transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform= transforms.ToTensor())

# Loading Train and Test data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
  plt.subplot(2,3,i+1)
  plt.imshow(samples[i][0], cmap='gray')
#plt.show() plotting image on the screen
# get image on tensorboad
im_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_samples', im_grid)
writer.close()

#Building the model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Model graph
writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()


# Train the Model
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0


for epoch in range(num_epochs):
  # loop through batches
  for i, (images, labels) in enumerate(train_loader):
    # 100, 1,28,28
    # 100, 784
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    # Forward
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backwards
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # statistics
    running_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    running_correct += (predicted == labels).sum().item()

    if (i+1) % 100 == 0:
      print("Epoch {}/{}, Step {}/{}, Loss: {:.4f}".format(epoch+1, num_epochs, i+1, n_total_steps, loss.item()))
      writer.add_scalar('training/loss', running_loss/100, epoch*n_total_steps + i)
      writer.add_scalar('accuracy', running_correct/100, epoch*n_total_steps + i)
      running_loss = 0.0
      running_correct = 0

# Test the Model 
labels = []
preds = [] 
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for images, labels1 in test_loader:
    images = images.reshape(-1,28*28).to(device)
    labels1 = labels1.to(device)
    outputs = model(images)

    _, predictions = torch.max(outputs,1)
    n_samples += labels1.shape[0]
    n_correct += (predictions == labels1).sum().item()

    class_preds = [F.softmax(output, dim=0) for output in outputs]

    labels.append(predictions)
    preds.append(class_preds)
    
  labels = torch.cat(labels)
  preds = torch.cat([torch.stack(batch) for batch in preds])

  acc = 100.0* n_correct / n_samples
  print("Accuracy = {}%".format(acc))

  classes = range(10)
  for i in range(10):
    label_i = labels == i
    pred_i = preds[:,i]
    writer.add_pr_curve('pr_curve_{}'.format(i), label_i, pred_i, global_step=0)
    writer.close()
# Save the Model
torch.save(model.state_dict(), 'model.ckpt')

# Load the Model
model.load_state_dict(torch.load('model.ckpt'))