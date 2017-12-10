# -*- coding: utf-8 -*-
"""
#############pytorch强化学习DQN 之Qlearning############
Cartpole模型说明:
#观察值:Observation,观察值包含均布的随机值噪声+-0.5
0:小车位置  -2.4~2.4
1:小车速度 -Inf~Inf
2:摆角度   -41.8~41.8
3:摆角速度 -Inf~Inf
#行为:Action
0: 小车向左
1: 小车向右
#仿真结束条件:
1.摆的角度不在+-12°
2.小车的单位不在+-2.4
3.episode 长度大于200
关于本例中state,action,next_state,reward
state:为当前图片-上步的图片,差值可以考虑到速度
action:shape=(1,1)的tensor,取值时需要action[0][0]
next_state:下步的状态
reward:默认的reward都是1,停止也是1
"""

import math
import matplotlib
import matplotlib.pyplot as plt
import random
from PIL import Image
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

# 创建倒立摆的环境,unwrapped可以获得环境内部参数,更方便进行深层次的操作
# 如果上传代码到openai,必须去除这个函数,防作弊
env = gym.make('CartPole-v0').unwrapped
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()  # 开启交互模式,该模式下可显示多个图片(非阻塞),但只有调用plt.show()才会显示图片

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

# 当前状态/行为/下个状态/收益
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # 超过capacity则循环重头开始覆盖
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        "从下步可用状态中随机抽取N个"
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# Now, let's define our model. But first, let quickly recap what a DQN is.
#
# DQN algorithm
# -------------
#
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
#
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
# :math:`R_{t_0}` is also known as the *return*. The discount,
# :math:`\gamma`, should be a constant between :math:`0` and :math:`1`
# that ensures the sum converges. It makes rewards from the uncertain far
# future less important for our agent than the ones in the near future
# that it can be fairly confident about.
#
# The main idea behind Q-learning is that if we had a function
# :math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
#
# .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
#
# However, we don't know everything about the world, so we don't have
# access to :math:`Q^*`. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# :math:`Q^*`.
#
# For our training update rule, we'll use a fact that every :math:`Q`
# function for some policy obeys the Bellman equation:
#
# .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
#
# The difference between the two sides of the equality is known as the
# temporal difference error, :math:`\delta`:
#
# .. math:: \delta = Q(s, a) - (r + \gamma \max_a Q(s', a))
#
# To minimise this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of :math:`Q` are very noisy. We calculate
# this over a batch of transitions, :math:`B`, sampled from the replay
# memory:
#
# .. math::
#
#    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta) 
#
# .. math::
#
#    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}
#
# Q-network
# ^^^^^^^^^
#
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing :math:`Q(s, \mathrm{left})` and
# :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# network). In effect, the network is trying to predict the *quality* of
# taking each action given the current input.
#


class DQN(nn.Module):
    '''输入为state,为2副相邻图像的插值,shape=1,3,40,80
       输出为行为(0,1)的对应概率,代表左和右,bath_size=1,channel=3
       input=3,output=2
    '''

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


######################################################################
# Input extraction
# ^^^^^^^^^^^^^^^^
#
# The code below are utilities for extracting and processing rendered
# images from the environment. It uses the ``torchvision`` package, which
# makes it easy to compose image transforms. Once you run the cell it will
# display an example patch that it extracted.
#

# 转换获得的图片(ndarray)->3*320*160
# 转换为PIL->缩小尺寸为80x40>转换为Tensor
resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# This is based on the code from gym.
screen_width = 600


# 获得小车当前state中心的像素位置
def get_cart_location():
    # env.x_threshold小车的X单向最大范围
    world_width = env.x_threshold * 2
    # 单个像素对应的距离
    scale = screen_width / world_width
    # state=x,speed_x,a,speed_a,state[0]=当前小车的x位置,起点为左侧边缘
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


# 采集小车当前状态下的图片并裁剪大小为320*160,转换为(1,3,40,80)的Tensor
def get_screen():
    # 返回numpy_array,由于ndarray的分布为(row,clolumn,channel),需要转换为(channel,row,clolumn)
    # 对于该图片,由(400,600,3)->(3,400,600),pytorch的格式为CHW
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # 高度方向裁剪,只保留图片高度方向的中间部分,(3,400,600)->(3,160,600)
    screen = screen[:, 160:320]
    view_width = 320  # 视野宽度
    cart_location = get_cart_location()
    # //当0<x<160,则mask=(0,320)
    if cart_location < view_width // 2:
        # 等价于[0,view_width]
        slice_range = slice(view_width)
    # //当x>600-160,则mask=(-320,0)
    elif cart_location > (screen_width - view_width // 2):
        # 等价于[-view_width,0]
        slice_range = slice(-view_width, None)
    else:
        ##x为中心宽度为view_width
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # 水平方向裁剪,(3,160,600)->(3,160,320)
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # 转换为C内存连续空间变量,并归一化
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # 转换为torch 的Tensor
    screen = torch.from_numpy(screen)
    # 转换为PIL Image,缩小尺寸,添加bath维度,转换为Tensor,(3,160,320)->(1,3,40,80)
    return resize(screen).unsqueeze(0).type(Tensor)


# 重置环境
env.reset()
plt.figure()
# squeeze降维->permute换位->转换为numpy
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``Variable`` - this is a simple wrapper around
#    ``torch.autograd.Variable`` that will automatically send the data to54
#    the GPU every time we construct a Variable.
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05  # 随机选择的
EPS_DECAY = 200  # 衰减终止迭代次数

model = DQN()

if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)
steps_done = 0


# 前200次eps_threshold较大,因此探索几率较多,随着不断的迭代
# eps_threshold呈指数衰减,exp(-1)=0.368,exp(-2)=0.135,exp(-5)=0.006
# 这意味着当steps_done=1000次时,eps_threshold=0.05+exp(-5)=0.00557,即5.57%
# 的几率进行探索,而其他时刻都只进行开发,state.shape=(1,3,40,80),输出为shape=(1,1)的
# action,可选值为[[0]]或者[[1]],
def select_action(state):
    global steps_done
    # 生成随机数
    sample = random.random()
    # 随机选择下步行为的阈值
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # 如果随机数大于阈值(迭代次数增多),则进行开发
    if sample > eps_threshold:
        # 将当前模型输入到model,输出(1,2)的action概率,求出最大概率的index,然后转换为(1,1)的tensor
        # max(1)返回最大值的同时还返回该最大值的位置index,max(1)[1]代表列最大值所在的行的index
        return model(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        # 随机探索
        return LongTensor([[random.randrange(2)]])


# 每轮仿真的步数,len=num_episodes
episode_durations = []


def plot_durations():
    plt.figure(1)  # figure id=2,如果存在则返回,不存在则创建
    plt.clf()  # 清除图像
    durations_t = torch.FloatTensor(episode_durations)  # 转换为tensor,shape=num_episodes
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    # 绘制本轮仿真经历了多少步,步数越多算法越好
    plt.plot(durations_t.numpy())
    # 从100轮开始,开始绘制本轮前100轮的平均值(包括本轮)
    if len(durations_t) >= 100:
        # 将tensor在指定维度展开,差分步为1,shape=[(n-100)%step+1,100],然后求每行的均值
        # 例如tensor第1行表示(1,100)轮的迭代次数,第2行表示(2,101)的迭代次数
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        # 默认在rows方向增加,前99轮不计平均值为,默认都为0.0
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # 暂停等待图像更新
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state.


last_sync = 0


def optimize_model():
    '''优化Qtable'''
    global last_sync
    # 如果memory中存的对象小于1个BATCH_SIZE,则退出
    if len(memory) < BATCH_SIZE:
        return
    # 从memory随机选择N个转换
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    # 将BATCH_SIZE个Transition转换为1个Transition
    # bach的每个参数都为128的元组
    batch = Transition(*zip(*transitions))

    # 计算当前Batch中的每个转换的下步状态部不为None的Index(1,0,1,...)
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    # 计算当前Batch中的每个转换的下步状态中非None的值,转换为n*1的shape的Variable
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]), volatile=True)
    # 将state/action/reward转换为Variable
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    # 神经网络计算Q(s_t),然后根据对应的ation选取对应的概率
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.

    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))

    # 神经网络计算Q(s_t+1),下步状态(非None)的输出的行为(0,1)的最大概率
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False

    # 期望的Q值,Q=Q(s',t')*GAMMA+reward
    # 等于当前(状态,行为)的reward+下步(状态,行为)的最大Q值*GAMMA
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算标准差为1的Huber loss,该方法对离群点不敏感,比square loss更具备鲁棒性
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        # 限制参数范围上下限为(-1,1)
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` variable. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes.

num_episodes = 10000
# 仿真100轮
for i_episode in range(num_episodes):
    # 每轮开始先初始化仿真环境
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    # 循环直至本次轮仿真结束,即done=true
    # 不用while的原因是count()可以生成迭代次数t,不用t+=1
    for t in count():
        # 选择action
        action = select_action(state)
        # env执行action,action=[[0]]或者[[1]]
        status, reward, done, _ = env.step(action[0, 0])
        x, vel_x, a, vel_a = status
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(a)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        # 将reward转换为tensor,shape=(1)
        reward = Tensor([reward])

        # 上步采集和当前采集的图像
        last_screen = current_screen
        current_screen = get_screen()
        # 如果仿真没有结束,下步的状态等于(当前状态-上步状态)
        if not done:
            next_state = current_screen - last_screen
        else:
            # 仿真结束,则下步为None
            next_state = None

        # 将转换存入memory
        memory.push(state, action, next_state, reward)

        # 移动到下一步状态
        state = next_state

        # 优化QTable
        optimize_model()
        # 如果本轮仿真结束,同时记录本轮仿真的最终步数
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
# 关闭渲染
env.render(close=True)
env.close()
plt.ioff()  # 关闭交互模式
plt.show()  # 显示图片
