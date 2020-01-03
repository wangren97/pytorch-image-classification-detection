import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])
##### CIFAR10: 3x32x32; 6000 images per class; 10 classes; 50000 + 10000 #####
cifar10_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
print('CIFAR10 Train')
print(cifar10_train_set.data.mean(axis=(0, 1, 2)) / 255)
print(cifar10_train_set.data.std(axis=(0, 1, 2)) / 255)
print('CIFAR10 Test')
print(cifar10_test_set.data.mean(axis=(0, 1, 2)) / 255)
print(cifar10_test_set.data.std(axis=(0, 1, 2)) / 255)

##### CIFAR100: 3x32x32; 600 images per class; 100 classes; 50000 + 10000 #####
cifar100_train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar100_test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
print('CIFAR100 Train')
print(cifar100_train_set.data.mean(axis=(0, 1, 2)) / 255)
print(cifar100_train_set.data.std(axis=(0, 1, 2)) / 255)
print('CIFAR100 Test')
print(cifar100_test_set.data.mean(axis=(0, 1, 2)) / 255)
print(cifar100_test_set.data.std(axis=(0, 1, 2)) / 255)
