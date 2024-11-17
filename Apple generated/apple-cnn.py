import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from torchvision import datasets, utils

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义超参数
batch_size = 64
latent_dim = 100
num_epochs = 2000
learning_rate = 0.0001


# 数据集准备
transform = transforms.Compose([
    # transforms.Resize(image_size),
    # transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),  # 输出为 [batch_size, 32, 52, 64]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 输出为 [batch_size, 64, 26, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 输出为 [batch_size, 128, 13, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),  # 展平为 [batch_size, 128 * 13 * 16]

            nn.Linear(128 * 13 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
class Generator(nn.Module):
    def __init__(self, latent_dim, channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128 * 13 * 16),  # 输出为 [batch_size, 128 * 13 * 16]
            nn.BatchNorm1d(128 * 13 * 16),
            nn.ReLU(True),

            nn.Unflatten(1, (128, 16, 13)),  # 转换为 [batch_size, 128, 13, 16]

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 输出为 [batch_size, 64, 26, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 输出为 [batch_size, 32, 52, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),  # 输出为 [batch_size, 3, 104, 128]
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
    
train_dataset = datasets.ImageFolder(root=r'E:\DEPP_RL\GAN\apple1', transform=transform)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 创建生成器和判别器
G = Generator(latent_dim=latent_dim).to(device)
D = Discriminator().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate)

# 创建用于保存图片的文件夹
output_dir = r'E:\DEPP_RL\GAN\Gapple1'
os.makedirs(output_dir, exist_ok=True)

# 开始训练
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # 转换数据到设备
        images = images.to(device)
        
        # 训练判别器

        outputs_R= D(images)
        d_loss_real = criterion(outputs_R, real_labels)
        real_score = outputs_R
        
        # 生成假图像
        z = torch.randn(batch_size, latent_dim).to(device)  # 100 噪声
        fake_images = G(z)
        outputs_F= D(fake_images.detach())
  
        fake_score = outputs_F
        if fake_score.mean().item()>0.2:
        # 总的判别器损失
            D.zero_grad()
            d_loss_fake = criterion(outputs_F, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
        
        # 训练生成器
        G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # 生成器试图欺骗判别器
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()
        
        if  fake_score.mean().item()> 0.2:
            print(i)
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')
            
            # 生成固定的噪声向量，用于生成图像
            fixed_noise = torch.randn(batch_size, latent_dim).to(device)

            # 使用生成器生成一批图像
            generated_images = G(fixed_noise)

            # 保存图片到指定文件夹
            # 构建保存图像的文件名
            image_filename = f'{output_dir}/epoch_{epoch+1}_step_{i+1}.png'

            # 使用 torchvision.utils.make_grid 将多个图像组合成一个网格
            grid = utils.make_grid(generated_images.cpu(), nrow=8, normalize=True, pad_value=1)

            # 使用 torchvision.utils.save_image 保存组合后的图像到指定文件
            utils.save_image(grid, image_filename)

            # 输出保存图像的信息
            print(f'Saved images at epoch [{epoch+1}/{num_epochs}] step [{i+1}/{len(data_loader)}]')