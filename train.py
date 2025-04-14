import os
import argparse
import config
import torch
import yaml
from typing import Any
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from logging import Logger
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from network import discriminator
from network import generator
from torch import Tensor
from PIL import Image
from tqdm import tqdm
from dataset import *
from utils import *
import torch.nn as nn
import numpy as np
import itertools
import logging
import warnings

warnings.filterwarnings(action='ignore')

# 保存模型
def save_checkpoint(
        model: Module,
        optimeter: Adam,
        scheduler: LambdaLR,
        weight_file_name: str,
        args: Any,
) -> None:
    if not os.path.exists(f'{args["MODEL"]["G"]["NAME"]}_model'):
        os.mkdir(f'{args["MODEL"]["G"]["NAME"]}_model')
    torch.save({'model_state_dict': model.state_dict(),
                'optimeter_state_dict': optimeter.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()},
               os.path.join(f'{args["MODEL"]["G"]["NAME"]}_model', weight_file_name))


# 加载模型
def load_checkpoint(
        model: Module,
        optimeter: Adam,
        scheduler: LambdaLR,
        weight_path: str,
) -> [Module, Adam, LambdaLR]:
    checkpoint = torch.load(weight_path,
                            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimeter.load_state_dict(checkpoint['optimeter_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimeter, scheduler


# 加载训练数据集
def load_datasets(
        args: Any
) -> [DataLoader, DataLoader]:
    # 图像 transforms
    train_transforms = [
        transforms.Resize(int(args['TRAIN']['DATASET']['IMAGE_SIZE'] * 1.12), Image.BICUBIC),  # 图片放大1.12倍
        transforms.RandomCrop((args['TRAIN']['DATASET']['IMAGE_SIZE'], args['TRAIN']['DATASET']['IMAGE_SIZE'])),
        # 随机裁剪为原来的大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 变为Tensor数据，归一化到[0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化到[-1, 1]
    ]
    val_transforms = [
        transforms.ToTensor(),  # 变为Tensor数据，归一化到[0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化到[-1, 1]
    ]
    # Training data loader
    train_dataloader = DataLoader(
        ImageDataset(args['TRAIN']['DATASET']['TRAIN_DATASET_PATH'],
                     transforms_=train_transforms, unaligned=True),
        batch_size=args['TRAIN']['DL']['BATCH_SIZE'],
        shuffle=args['TRAIN']['DL']['SHUFFLE'],
        pin_memory=args['TRAIN']['DL']['PIN_MEMORY'],
        num_workers=args['TRAIN']['DL']['NUM_WORKERS'],
    )
    # Test data loader
    val_dataloader = DataLoader(
        ImageDataset(args['TRAIN']['DATASET']['TEST_DATASET_PATH'],
                     transforms_=val_transforms, unaligned=False),
        batch_size=4,
        shuffle=args['TRAIN']['DL']['SHUFFLE']
    )
    return train_dataloader, val_dataloader


# 定义模型
def build_model(
        args: Any,
        device: torch.device
) -> [Module, Module, Module, Module]:

    G_AB = generator.__dict__[args['MODEL']['G']['NAME']](input_dim=args['MODEL']['G']['IN_CHANNELS'],
                                                          encoder_layers=args['MODEL']['G']['ENCODER_LAYERS'],
                                                          num_residual_block=args['MODEL']['G']['NUM_RESIDUAL_BLOCK'])

    G_BA = generator.__dict__[args['MODEL']['G']['NAME']](input_dim=args['MODEL']['G']['IN_CHANNELS'],
                                                          encoder_layers=args['MODEL']['G']['ENCODER_LAYERS'],
                                                          num_residual_block=args['MODEL']['G']['NUM_RESIDUAL_BLOCK'])
    

    D_A = discriminator.__dict__[args['MODEL']['D']['NAME']](input_dim=args['MODEL']['D']['IN_CHANNELS'],
                                                             spectral_norm=args['MODEL']['D']['SPECTRAL_NORM'])

    D_B = discriminator.__dict__[args['MODEL']['D']['NAME']](input_dim=args['MODEL']['D']['IN_CHANNELS'],
                                                             spectral_norm=args['MODEL']['D']['SPECTRAL_NORM'])
    
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)

    return G_AB, G_BA, D_A, D_B


# 定义损失函数
def define_loss(
        args: Any,
        device: torch.device
) -> [nn.L1Loss, nn.MSELoss, nn.L1Loss, FocalFrequencyLoss]:
    identify_criterion = nn.L1Loss()
    gan_criterion = nn.MSELoss()
    cycle_criterion = nn.L1Loss()
    cycle_frequency_criterion = FocalFrequencyLoss(loss_weight=args['TRAIN']['LOSS']['FREQUENCY_LOSS_WEIGHT'], 
                                                   alpha=0.5, patch_factor=1)
    identify_criterion = identify_criterion.to(device)
    gan_criterion = gan_criterion.to(device)
    cycle_criterion = cycle_criterion.to(device)
    cycle_frequency_criterion = cycle_frequency_criterion.to(device)

    return identify_criterion, gan_criterion, cycle_criterion, cycle_frequency_criterion


# 定义优化器
def define_optimeter(
        G_AB: Module,
        G_BA: Module,
        D_A: Module,
        D_B: Module,
        args: Any,
) -> [Adam, Adam, Adam]:
    
    g_optimeter = Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                       lr=args['TRAIN']['OPTIM']['G_LR'],
                       betas=(args['TRAIN']['OPTIM']['B1'], args['TRAIN']['OPTIM']['B2']))
    
    d_A_optimeter = Adam(D_A.parameters(),
                         lr=args['TRAIN']['OPTIM']['D_LR'],
                         betas=(args['TRAIN']['OPTIM']['B1'], args['TRAIN']['OPTIM']['B2']))
    
    d_B_optimeter = Adam(D_B.parameters(),
                         lr=args['TRAIN']['OPTIM']['D_LR'],
                         betas=(args['TRAIN']['OPTIM']['B1'], args['TRAIN']['OPTIM']['B2']))

    return g_optimeter, d_A_optimeter, d_B_optimeter


# 定义退化学习率
def define_lr_scheduler(
        g_optimeter: Adam,
        d_A_optimeter: Adam,
        d_B_optimeter: Adam,
        args: Any
) -> [LambdaLR, LambdaLR, LambdaLR]:
    
    g_lr_scheduler = LambdaLR(
        optimizer=g_optimeter,
        lr_lambda=DecayLR(args['TRAIN']['LR']['N_EPOCHS'],
                          args['TRAIN']['LR']['EPOCH'],
                          args['TRAIN']['LR']['DECAY_EPOCH']).step
    )
    
    d_A_lr_scheduler = LambdaLR(
        optimizer=d_A_optimeter,
        lr_lambda=DecayLR(args['TRAIN']['LR']['N_EPOCHS'],
                          args['TRAIN']['LR']['EPOCH'],
                          args['TRAIN']['LR']['DECAY_EPOCH']).step
    )
    
    d_B_lr_scheduler = LambdaLR(
        optimizer=d_B_optimeter,
        lr_lambda=DecayLR(args['TRAIN']['LR']['N_EPOCHS'],
                          args['TRAIN']['LR']['EPOCH'],
                          args['TRAIN']['LR']['DECAY_EPOCH']).step
    )

    return g_lr_scheduler, d_A_lr_scheduler, d_B_lr_scheduler


# 验证模型
def eval(
        G_AB: Module,
        G_BA: Module,
        batches_done: int,
        val_dataloader: DataLoader,
        device: torch.device,
        args: Any,
        nrow: int = 4
) -> None:
    # ----------
    #  Valling
    # ----------
    imgs = iter(val_dataloader).__next__()  ## 取一张图像
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"], requires_grad=False)  # 取一张真A
    real_B = Variable(imgs["B"], requires_grad=False)  # 去一张真B
    real_A = real_A.to(device)
    real_B = real_B.to(device)
    fake_B = G_AB(real_A)  # 用真A生成假B
    fake_A = G_BA(real_B)  # 用真B生成假A
    # make_grid():用于把几个图像按照网格排列的方式绘制出来
    real_A = make_grid(real_A, nrow=nrow, normalize=True)
    real_B = make_grid(real_B, nrow=nrow, normalize=True)
    fake_A = make_grid(fake_A, nrow=nrow, normalize=True)
    fake_B = make_grid(fake_B, nrow=nrow, normalize=True)
    # 把以上图像都拼接起来，保存为一张大图片
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), dim=1)
    if not os.path.exists(f'{args["MODEL"]["G"]["NAME"]}_images'):
        os.mkdir(f'{args["MODEL"]["G"]["NAME"]}_images')
    save_image(image_grid, f'{args["MODEL"]["G"]["NAME"]}_images/{batches_done}.png', normalize=False)

# 训练模型
def train(
        G_AB: Module,
        G_BA: Module,
        D_A: Module,
        D_B: Module,
        train_dataloader: DataLoader,
        input_A: Tensor,
        input_B: Tensor,
        target_real: Tensor,
        target_fake: Tensor,
        identify_criterion: nn.L1Loss,
        gan_criterion: nn.MSELoss,
        cycle_criterion: nn.L1Loss,
        cycle_frequency_criterion: FocalFrequencyLoss,
        g_optimeter: Adam,
        d_A_optimeter: Adam,
        d_B_optimeter: Adam,
        fake_A_buffer: ReplayBuffer,
        fake_B_buffer: ReplayBuffer,
        epoch: int,
        writer: SummaryWriter,
        logger: Logger,
        device: torch.device,
        args: Any
) -> None:
    for i, batch in enumerate(tqdm(train_dataloader)):
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        """ Generators A2B and B2A """
        G_AB.train()
        G_BA.train()

        g_optimeter.zero_grad()

        # Identify loss
        same_B = G_AB(real_B)
        identify_loss_B = identify_criterion(same_B, real_B) * args['TRAIN']['LOSS']['IDENTIFY_WEIGHT']

        same_A = G_BA(real_A)
        identify_loss_A = identify_criterion(same_A, real_A) * args['TRAIN']['LOSS']['IDENTIFY_WEIGHT']

        identify_loss = identify_loss_B + identify_loss_A

        # GAN loss
        fake_B = G_AB(real_A)
        pred_fake = D_B(fake_B)
        gan_loss_A2B = gan_criterion(pred_fake, target_real)
        
        fake_A = G_BA(real_B)
        pred_fake = D_A(fake_A)
        gan_loss_B2A = gan_criterion(pred_fake, target_real)
        
        gan_loss_G = gan_loss_A2B + gan_loss_B2A

        # Cycle loss
        if args['TRAIN']['LOSS']['IS_FREQUENCY_LOSS']:
            recovered_A = G_BA(fake_B)
            cycle_loss_l1_ABA = cycle_criterion(recovered_A, real_A)
            cycle_loss_frequency_ABA = cycle_frequency_criterion(recovered_A, real_A)
            cycle_loss_ABA = (cycle_loss_l1_ABA + cycle_loss_frequency_ABA) * args['TRAIN']['LOSS']['CYCLE_WEIGHT']

            recovered_B = G_AB(fake_A)
            cycle_loss_l1_BAB = cycle_criterion(recovered_B, real_B)
            cycle_loss_frequency_BAB = cycle_frequency_criterion(recovered_B, real_B)
            cycle_loss_BAB = (cycle_loss_l1_BAB + cycle_loss_frequency_BAB) * args['TRAIN']['LOSS']['CYCLE_WEIGHT']

            cycle_loss = cycle_loss_ABA + cycle_loss_BAB
        else:
            recovered_A = G_BA(fake_B)
            cycle_loss_ABA = cycle_criterion(recovered_A, real_A) * args['TRAIN']['LOSS']['CYCLE_WEIGHT']

            recovered_B = G_AB(fake_A)
            cycle_loss_BAB = cycle_criterion(recovered_B, real_B) * args['TRAIN']['LOSS']['CYCLE_WEIGHT']

            cycle_loss = cycle_loss_ABA + cycle_loss_BAB


        # Total loss
        g_loss = identify_loss + gan_loss_G + cycle_loss

        g_loss.backward()
        g_optimeter.step()

        images = [real_A, fake_B, recovered_A, same_A, real_B, fake_A, recovered_B, same_B]

        """ Discriminator A"""
        D_A.train()
        
        d_A_optimeter.zero_grad()
        
        # Real loss
        pred_real = D_A(real_A)
        d_real_loss_A = gan_criterion(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = D_A(fake_A.detach())
        d_fake_loss_A = gan_criterion(pred_fake, target_fake)

        # Total loss
        d_loss_A = (d_real_loss_A + d_fake_loss_A) * 0.5
        
        d_loss_A.backward()
        d_A_optimeter.step()
        

        """ Discriminator B """
        D_B.train()
        
        d_B_optimeter.zero_grad()

        # Real loss
        pred_real = D_B(real_B)
        d_real_loss_B = gan_criterion(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = D_B(fake_B.detach())
        d_fake_loss_B = gan_criterion(pred_fake, target_fake)

        # Total loss
        d_loss_B = (d_real_loss_B + d_fake_loss_B) * 0.5
        
        d_loss_B.backward()
        d_B_optimeter.step()

        batches_done = epoch * len(train_dataloader) + (i + 1)

        if batches_done % int(len(train_dataloader) * 0.5) == 0:
            # 日志记录数据
            logger.info(
                f'epoch: {epoch}, batches_done: {batches_done},'
                f'identify_loss: {identify_loss.item()},'
                f'gan_loss_G: {gan_loss_G.item()},'
                f'cycle_loss_ABA: {cycle_loss_ABA.item()},'
                f'cycle_loss_BAB: {cycle_loss_BAB.item()},'
                f'cycle_loss: {cycle_loss.item()},'
                f'g_loss: {g_loss.item()},'
                f'd_loss_A: {d_loss_A.item()},'
                f'd_loss_B: {d_loss_B.item()},'
                f'd_loss: {(d_loss_A + d_loss_B).item()},'
                f'current_g_lr: {g_optimeter.state_dict()["param_groups"][0]["lr"]},'
                f'current_d_lr: {d_A_optimeter.state_dict()["param_groups"][0]["lr"]}'
            )
            images = list(map(lambda x: x.squeeze(dim=0), images))
            image_tensor = make_grid(images, nrow=4, normalize=True)
            # tensorboard可视化显示
            writer.add_image('image_tensor', image_tensor, batches_done)
            writer.add_scalar('identify_loss', identify_loss.item(), batches_done)
            writer.add_scalar('gan_loss_G', gan_loss_G.item(), batches_done)
            writer.add_scalar('cycle_loss_ABA', cycle_loss_ABA.item(), batches_done)
            writer.add_scalar('cycle_loss_BAB', cycle_loss_BAB.item(), batches_done)
            writer.add_scalar('cycle_loss', cycle_loss.item(), batches_done)
            writer.add_scalar('g_loss', g_loss.item(), batches_done)
            writer.add_scalar('d_loss_A', d_loss_A.item(), batches_done)
            writer.add_scalar('d_loss_B', d_loss_B.item(), batches_done)
            writer.add_scalar('d_loss', (d_loss_A + d_loss_B).item(), batches_done)
            writer.add_scalar('current_g_lr', g_optimeter.state_dict()["param_groups"][0]["lr"], batches_done)
            writer.add_scalar('current_d_lr', d_A_optimeter.state_dict()["param_groups"][0]["lr"], batches_done)

# main
def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备信息：{device}')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./config/cyclegan.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    
    input_A = Tensor(config['TRAIN']['DL']['BATCH_SIZE'], config['MODEL']['G']['IN_CHANNELS'],
                     config['TRAIN']['DATASET']['IMAGE_SIZE'], config['TRAIN']['DATASET']['IMAGE_SIZE'])

    input_B = Tensor(config['TRAIN']['DL']['BATCH_SIZE'], config['MODEL']['G']['IN_CHANNELS'],
                     config['TRAIN']['DATASET']['IMAGE_SIZE'], config['TRAIN']['DATASET']['IMAGE_SIZE'])
    
    target_real = Variable(Tensor(config['TRAIN']['DL']['BATCH_SIZE'], 1,
                                  config['MODEL']['D']['OUT_SIZE'], config['MODEL']['D']['OUT_SIZE']).fill_(
        1.0), requires_grad=False)  # 全填充为1
    
    target_fake = Variable(Tensor(config['TRAIN']['DL']['BATCH_SIZE'], 1,
                                  config['MODEL']['D']['OUT_SIZE'], config['MODEL']['D']['OUT_SIZE']).fill_(
        0.0), requires_grad=False)  # 全填充为0
    
    train_dataloader, val_dataloader = load_datasets(config)

    G_AB, G_BA, D_A, D_B = build_model(config, device=device)

    identify_criterion, gan_criterion, cycle_criterion, cycle_frequency_criterion = define_loss(config, device)

    g_optimeter, d_A_optimeter, d_B_optimeter = define_optimeter(G_AB, G_BA, D_A, D_B, config)

    g_lr_scheduler, d_A_lr_scheduler, d_B_lr_scheduler = define_lr_scheduler(g_optimeter,
                                                                             d_A_optimeter, d_B_optimeter, config)
    
    # 读取模型
    if config['TRAIN']['LR']['EPOCH'] != 0:
        G_AB, g_optimeter, g_lr_scheduler = load_checkpoint(
            model=G_AB, optimeter=g_optimeter, scheduler=g_lr_scheduler,
            weight_path=f'{config["MODEL"]["G"]["NAME"]}_model/G_AB_{config["TRAIN"]["LR"]["EPOCH"]}.pth'
        )

        G_BA, g_optimeter, g_lr_scheduler = load_checkpoint(
            model=G_BA, optimeter=g_optimeter, scheduler=g_lr_scheduler,
            weight_path=f'{config["MODEL"]["G"]["NAME"]}_model/G_BA_{config["TRAIN"]["LR"]["EPOCH"]}.pth'
        )

        D_A, d_A_optimeter, d_A_lr_scheduler = load_checkpoint(
            model=D_A, optimeter=d_A_optimeter, scheduler=d_A_lr_scheduler,
            weight_path=f'{config["MODEL"]["G"]["NAME"]}_model/D_A_{config["TRAIN"]["LR"]["EPOCH"]}.pth'
        )

        D_B, d_B_optimeter, d_B_lr_scheduler = load_checkpoint(
            model=D_B, optimeter=d_B_optimeter, scheduler=d_B_lr_scheduler,
            weight_path=f'{config["MODEL"]["G"]["NAME"]}_model/D_B_{config["TRAIN"]["LR"]["EPOCH"]}.pth'
        )
        
    # 样本缓冲区
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # 初始化TensorBoard
    writer = SummaryWriter(f'{config["MODEL"]["G"]["NAME"]}_logs')
    # 初始化日志
    logging = logger(f'{config["MODEL"]["G"]["NAME"]}_run')

    """ Training """
    try:
        for epoch in range(config["TRAIN"]["LR"]["EPOCH"], config["TRAIN"]["LR"]["N_EPOCHS"]):
            # 训练模型
            train(G_AB, G_BA, D_A, D_B, train_dataloader,
                input_A, input_B, target_real, target_fake,
                identify_criterion, gan_criterion, cycle_criterion,
                cycle_frequency_criterion,
                g_optimeter, d_A_optimeter, d_B_optimeter,
                fake_A_buffer, fake_B_buffer, epoch, writer,
                logging, device, config)    
    
            # 学习率更新
            g_lr_scheduler.step()
            d_A_lr_scheduler.step()
            d_B_lr_scheduler.step()
    
            # 每10个epoch保存一次模型&测试模型
            if epoch % 10 == 0:
                eval(G_AB, G_BA, epoch, val_dataloader, device, config)
                save_checkpoint(G_AB, g_optimeter, g_lr_scheduler, f'G_AB_{epoch}.pth', config)
                save_checkpoint(G_BA, g_optimeter, g_lr_scheduler, f'G_BA_{epoch}.pth', config)
                save_checkpoint(D_A, d_A_optimeter, d_A_lr_scheduler, f'D_A_{epoch}.pth', config)
                save_checkpoint(D_B, d_B_optimeter, d_B_lr_scheduler, f'D_B_{epoch}.pth', config)   
    except torch.cuda.OutOfMemoryError:
        print('CUDA out of memory. Tried to allocate 16.00 MiB (GPU 0; 2.00 GiB total capacity; 1.53 GiB already allocated; 0 bytes free; 1.73 GiB reserved in total by PyTorch)')
       
if __name__ == '__main__':
    main()
