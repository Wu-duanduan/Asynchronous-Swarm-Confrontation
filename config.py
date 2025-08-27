#!/usr/bin/python

"""配置参数文件"""
class Config:
    def __init__(self):
        self.obs_dim = 40            # 单个智能体观测维度
        self.act_dim = 3             # 每个智能体的动作维度（调度、逃跑、追击）
        self.hidden_dim = 256
        self.n_agents = 5
        
        self.MAX_EPISODE = 100
        self.MAX_STEP = 300


        # --- 调整训练参数 ---
        self.batch_size = 512        # 增大批次应对联合轨迹数据
        self.buffer_size = 50000     # 经验池扩容
        self.minimal_size = 1000     # 最小启动训练样本量
           
        self.lr = 3e-4               # 学习率
        self.gamma = 0.95            # 降低折扣因子
        self.epsilon = 0.8           # 初始探索率（ε值）
        self.epsilon_decay = 0.97         # 每episode衰减率
        self.epsilon_min = 0.15           # 最小探索率（目前参数较大）

        self.explore_phase = 2000       # 完全随机探索阶段步数
        self.epsilon_init = 0.8         # 初始ε值
        self.epsilon_decay = 0.9995     # 衰减率
        self.epsilon_min = 0.05         # 最小探索率
        self.noise_scale_init = 0.5     # 初始噪声方差
        self.noise_min = 0.1            # 最小噪声强度

        self.target_update_interval = 10  # 每10 episode更新一次