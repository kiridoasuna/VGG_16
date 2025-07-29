# -*- coding: UTF-8 -*-
'''
@Project ：VGG_16 
@File    ：model_train.py
@Author  ：公众号：思维侣行
@Date    ：2025/7/23 15:04 
'''
import copy
import time
import pandas as pd

import torch
from scipy.stats import moment
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision import transforms
import torch.optim as optim
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from model import VGG

def data_process():
    """
    处理数据
    :return: 训练数据和验证数据
    """
    transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
    ])

    data = CIFAR10(root='./data',
                        train=True,
                        download=True,
                        transform=transform)
    train_data_size = round(len(data) * 0.8)
    val_data_size = len(data) - train_data_size

    train_data, val_data = random_split(data, [train_data_size, val_data_size])

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=3)

    val_data_loader = DataLoader(dataset=val_data,
                                 batch_size=128,
                                 shuffle=False,
                                 num_workers=3)

    return train_data_loader, val_data_loader

def train_process(model, train_data_loader, val_data_loader, epochs):
    # 初始化设备
    device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 优化器
    optimizer = SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60],
        gamma=0.1
    )
    # 交叉熵损失函数
    loss_func_cross_entropy = nn.CrossEntropyLoss()
    # 将模型放入设备中
    model = model.to(device_cuda)
    #保存模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 最高精度
    best_acc = 0
    # 使用下面数组收集每一次训练和验证
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    # 开始时间
    start_time = time.time()

    print(f"{'-'* 10} 训练开始 {'-'* 10}")
    for epoch in range(epochs):
        print(f"{epoch+1}/{epochs}")
        print('-'* 10)

        train_loss = 0
        train_correct_num = 0
        val_loss = 0
        val_correct_num = 0
        train_num = 0
        val_num = 0
        for batch_step, (b_x, b_y) in enumerate(train_data_loader):
            b_x, b_y = b_x.to(device_cuda), b_y.to(device_cuda)

            model.train()

            output = model(b_x)
            # 得到预测的分类
            pre_lab = torch.argmax(output, dim=1)
            loss = loss_func_cross_entropy(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_correct_num += torch.sum(pre_lab == b_y.detach())
            train_num += b_x.size(0)

        model.eval()
        with torch.no_grad():
            for batch_step, (val_x, val_y) in enumerate(val_data_loader):
                val_x, val_y = val_x.to(device_cuda), val_y.to(device_cuda)
                output = model(val_x)
                pre_val_lab = torch.argmax(output, dim=1)
                loss= loss_func_cross_entropy(output, val_y)

                val_loss += loss.item() * val_x.size(0)
                val_correct_num += torch.sum(pre_val_lab == val_y.detach())
                val_num += val_x.size(0)

        # 统计每一次训练的所有数据
        #训练集
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_correct_num.double().item() / train_num )

        # 验证集
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correct_num.double().item() / val_num)

        print(f"train_loss: {train_loss_all[-1]}, train_acc: {train_acc_all[-1]}")
        print(f"val_loss: {val_loss_all[-1]}, val_acc: {val_acc_all[-1]}")

        lr_scheduler.step()

        # 保存最优精度的模型
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        # 计算训练和验证的时间
        cost_time = time.time() - start_time
        print(f"训练和验证耗费的时间{cost_time // 60:.0f}m {cost_time % 60:.0f}s ")

    torch.save(best_model_wts, './vgg_model/best_model.pth')

    train_process_data = pd.DataFrame(data={"epoch": range(1, epochs+1),
                                            "train_loss_all":train_loss_all,
                                            "val_loss_all":val_loss_all,
                                            "train_acc_all":train_acc_all,
                                            "val_acc_all":val_acc_all,})

    return train_process_data


def show_acc_loss_matplot(data):
    # 输入中文和符号显示不正常
    # plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei Mono"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

    # 如果中文问题解决不了的话，使用下面的方式给图中每一个使用中文的属性设定字体
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    font_property = FontProperties(fname=font_path)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    plt.plot(data["epoch"], data["train_loss_all"], "ro-", label="训练-损失值")
    plt.plot(data["epoch"], data["val_loss_all"], "bo-", label="验证-损失值")
    plt.legend(prop=font_property)
    plt.grid(True)
    plt.xlabel("训练次数", fontproperties=font_property)
    plt.ylabel("损失值", fontproperties=font_property)
    plt.title("损失变化曲线", fontproperties=font_property)

    plt.subplot(1, 2, 2)

    plt.plot(data["epoch"], data["train_acc_all"], "ro-", label="训练-准确度")
    plt.plot(data["epoch"], data["val_acc_all"], "bo-", label="验证-准确度")
    plt.legend(prop=font_property)
    plt.grid(True)
    plt.xlabel("训练次数", fontproperties=font_property)
    plt.ylabel("准确度", fontproperties=font_property)
    plt.title("准确度变化曲线", fontproperties=font_property)

    plt.show()

if __name__ == '__main__':
    vgg = VGG()

    train_data, val_data = data_process()

    train_process_data = train_process(vgg, train_data, val_data, 74)

    show_acc_loss_matplot(train_process_data)
