# 导入PyTorch库和其他所需库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
from collections import deque
import numpy as np
import cv2
from yaml import *

# 定义神经网络模型
class NET(nn.Module):
    # 初始化网络结构
    def __init__(self, observation_height, observation_width, action_space) -> None:
        super(NET, self).__init__()  # 调用父类构造函数
        self.state_dim = observation_width * observation_height  # 输入的特征维度
        self.state_w = observation_width  # 输入宽度
        self.state_h = observation_height  # 输入高度
        self.action_dim = action_space  # 动作空间大小
        self.relu = nn.ReLU()  # 定义ReLU激活函数
        # 卷积层定义
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=[5,5], stride=1, padding='same'),  # 第一层卷积
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层
            nn.Conv2d(32, 64, kernel_size=[5,5], stride=1, padding='same'),  # 第二层卷积
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        )
        
         # 全连接层定义a
        
        self.fc1 = nn.Linear(int((self.state_w/4) * (self.state_h/4) * 64), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.action_dim)

    # 前向传播函数
    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x =torch.tensor(x, dtype=torch.float, device=mps)
        x = self.net(x)  # 通过卷积层
        x = x.view(-1, int((self.state_w/4) * (self.state_h/4) * 64))  # 展平
        x = self.relu(self.fc1(x))  # 第一个全连接层
        x = self.relu(self.fc2(x))  # 第二个全连接层
        x = self.fc3(x)  # 输出层
        return x  # 返回输出
    
# 定义DQN类
class DQN(object):
    # 初始化DQN对象
    def __init__(self, observation_height, observation_width, action_space, model_file, log_file):
        self.model_file = model_file  # 模型文件路径
        self.target_net = NET(observation_height, observation_width, action_space)  # 目标网络
        self.target_net.to(mps)  # 将网络放到指定设备上运行
        self.eval_net = NET(observation_height, observation_width, action_space)  # 评估网络
        self.eval_net.to(mps)  # 将网络放到指定设备上运行
        self.eval_net.load_state_dict(self.target_net.state_dict())
        self.replay_buffer = deque()  # 经验回放池
        self.epsilon = INITIAL_EPSILON  # ε值
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)  # 优化器
        self.loss = nn.MSELoss()  # 损失函数
        self.action_dim = action_space  # 动作空间大小

    # 根据ε-贪心策略选择动作
    def choose_action(self, state):
        if random.random() <= self.epsilon:  # 随机选择动作
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000  # 减小ε值
            return random.randint(0, self.action_dim - 1)  # 返回随机动作索引
        else:  # 根据网络输出选择最优动作
            Q_value = self.eval_net(state)  # 获取Q值
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000  # 减小ε值10000次
            return torch.argmax(Q_value)  # 返回最大Q值对应的动作索引


    def store_data(self, state, action, reward, next_state, done):
        # 确保状态和下一个状态为 float 类型，并且形状正确
        
        next_state =torch.tensor(next_state, dtype=torch.float, device=mps)
        state =torch.tensor(state, dtype=torch.float, device=mps)
        # 如果状态的形状不正确，调整形状
        if len(state.shape) == 3:
            state = state.unsqueeze(0)  # 添加 batch 维度
        if len(state.shape) == 4 and state.shape[1] != 1:
            state = state.permute(0, 3, 1, 2)  # 调整通道维度
        
        if len(next_state.shape) == 3:
            next_state = next_state.unsqueeze(0)  # 添加 batch 维度
        if len(next_state.shape) == 4 and next_state.shape[1] != 1:
            next_state = next_state.permute(0, 3, 1, 2)  # 调整通道维度
        
        # # 将动作、奖励和完成标志转换为张量
        action = torch.tensor([action], dtype=torch.float, device=mps)  #one-hot ,适用与复杂维度，占内存
        reward = torch.tensor([reward], dtype=torch.float, device=mps)
        done = torch.tensor([done], dtype=torch.float, device=mps)
        
        # 存储数据
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    # 如果经验池已满
        if len(self.replay_buffer) > REPLAY_SIZE:
         self.replay_buffer.popleft()  # 移除最老的经验f
         
         
         
    def train(self):
        if len(self.replay_buffer) > BATCH_SIZE:
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)
            # 将状态列表转换为单个张量，并确保为 float 类型
            state_batch_tensor = torch.cat(state_batch).float()
            next_state_batch_tensor = torch.cat(next_state_batch).float()
            state_batch_tensor = state_batch_tensor.to(mps)
            next_state_batch_tensor = next_state_batch_tensor.to(mps)

            # 将动作、奖励和完成标志转换为张量
            action_batch_tensor = torch.cat(action_batch).to(mps)
            # 确保动作张量为 int64 类型
            action_batch_tensor = action_batch_tensor.long()
            reward_batch_tensor = torch.tensor(reward_batch, dtype=torch.float, device=mps)
            done_batch_tensor = torch.tensor(done_batch, dtype=torch.float, device=mps)

            # 计算目标网络的Q值
            Q_next = self.target_net(next_state_batch_tensor).max(1)[0].detach()

            # 计算目标Q值
            Q_target = reward_batch_tensor + (1 - done_batch_tensor) * GAMMA * Q_next

            # 计算评估网络的Q值
            Q_eval = self.eval_net(state_batch_tensor).gather(1, action_batch_tensor.unsqueeze(1)).squeeze(1)

            # 计算损失
            loss = self.loss(Q_eval, Q_target)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

    # 保存模型
    def save_model(self):
        torch.save(self.target_net.state_dict(), self.model_file)  # 保存模型参数

    # 同步评估网络到目标网络
    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())  # 加载评估网络的参数到目标网络






       

        