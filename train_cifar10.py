import time
import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from vgg_16 import VGG

##### Prepare for dataset #####
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])])

valid_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])])

test_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.49421428, 0.48513139, 0.45040909], std=[0.24665252, 0.24289226, 0.26159238])])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

valid_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=valid_transform)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=True, num_workers=0, pin_memory=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=0, pin_memory=True)
##### Define GPU environment; Model; Loss function; Optimizer; #####
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed(100)

model = VGG('VGG16')
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
cudnn.benchmark = True
best_accuracy = 0
##### Function for adjusting learning rate #####
def lr_adjust(optimizer, epoch):
    if epoch < 150:
        lr = 0.1
    elif epoch < 300:
        lr = 0.01
    elif epoch < 450:
        lr = 0.001

    for item in optimizer.param_groups:
        item['lr'] = lr

##### Function for training #####
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        temp, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    print('Epoch[%d] ---> Train Accuracy: %.5f%% (%d/%d) %.5f' % (epoch + 1, 100.*correct/total, correct, total, train_loss))

##### Function for test #####
def test(epoch):
    global best_accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            temp, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Epoch[%d] ---> Test Accuracy: %.3f%%  (%d/%d)' % (epoch + 1, 100.*correct/total, correct, total))

    accuracy = 100.*correct/total
    if accuracy > best_accuracy:
        print('Saving weight file')
        torch.save(model.state_dict(), './weights/vgg16_cifar10_{}.pth'.format(epoch))
        best_accuracy = accuracy

##### Main function #####
previous_time = time.time()
if __name__ == '__main__':
    # train
    
    for epoch in range(0, 450):
        lr_adjust(optimizer, epoch)
        train(epoch)
        test(epoch)
        current_time = time.time()
        time_left = datetime.timedelta(seconds=(current_time - previous_time)*(449 - epoch))
        previous_time = current_time
        print(time_left)
    
    # test
    ''' 
    model.load_state_dict(torch.load('./weights/vgg16_cifar10_360.pth'))
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            temp, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    end_time = time.time()
    time_cost = datetime.timedelta(seconds=(end_time - start_time))
    print(time_cost)
    print('Best Test Accuracy: %.3f%%  (%d/%d)' % (100.*correct/total, correct, total))
    '''
