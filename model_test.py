# -*- coding: UTF-8 -*-
'''
@Project ：VGG_16 
@File    ：model_test.py.py
@Author  ：公众号：思维侣行
@Date    ：2025/7/23 15:04 
'''
import torch
import torch.nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import VGG


def test_data_process():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    test_data = FashionMNIST(root='./data',
                             train=False,
                             download=True,
                             transform=transform)

    test_data_loader = DataLoader(dataset=test_data,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  )

    return test_data_loader


def test_process(model, test_data):
    # 数据准备
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(test_device)

    total_num = len(test_data)
    correct_num = 0

    with torch.no_grad():
        model.eval()
    for (x, y) in test_data:
        test_x, test_y = x.to(test_device), y.to(test_device)
        output = model(test_x)
        pre_lab = torch.argmax(output, dim=1)
        correct_num += torch.sum(pre_lab == test_y.detach())

    test_acc = correct_num.double().item() / total_num
    print(f"准确度是{test_acc}")


if __name__ == '__main__':
    vgg = VGG()
    vgg.load_state_dict(torch.load("./vgg_model/best_model.pth"))

    test_data = test_data_process()

    test_process(vgg, test_data)