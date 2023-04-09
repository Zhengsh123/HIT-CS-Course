
import argparse
import os
import torch
from net.optim.config import OptimConfig
from net.util import train,test
from generate_data import generate_train_data,generate_test_data,MyTestDataset,MyTrainDataset
from net.optim.model import InceptionResnetV2
from net.VGG.model import VGG
from net.VGG.config import VggConfig
from net.SE_ResNet.resnet_model import SEResNet18
from net.SE_ResNet.config import SEConfig
from net.ResNet.model import ResNet18
from net.ResNet.config import resnet_Config
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', dest='epoch', type=int, default=15, help='# of epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
    parser.add_argument('--device', dest='device', type=int, default=0, help='gpu flag, 0 for GPU and 1 for CPU')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
    args = parser.parse_args()
    # 指定可用显卡，笔记本上只有一个独显，只支持输入为0
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # 网络定义，需要运行哪一个网络就注释其他网络即可
    # 模型：VGG
    # config=VggConfig(args)
    # model=VGG().to(config.device)
    # 模型：ResNet
    config=resnet_Config()
    model=ResNet18().to(config.device)
    # 模型：Se_ResNet
    # config=SEConfig(args)
    # model=SEResNet18().to(config.device)

    # 模型：自选优化，Inception ResNet V2
    # config = OptimConfig(args)
    # model = InceptionResnetV2(12).to(config.device)
    # 训练
    train_data, dev_data, label2class = generate_train_data(config.train_path)
    # train_data = MyTrainDataset(train_data, config)
    # dev_data = MyTrainDataset(dev_data, config)
    # train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    # dev_loader = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False)
    # train(config, model, train_loader, dev_loader)
    # torch.save(model.state_dict(), config.model_save_path)
    # 生成测试结果
    # 加载模型
    model.load_state_dict(torch.load(config.model_save_path))
    test_data, file_data = generate_test_data(config.test_path)
    test_data = MyTestDataset(test_data, config)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    test(config, model, test_loader, label2class, file_data)
