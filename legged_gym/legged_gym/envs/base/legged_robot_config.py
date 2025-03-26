# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096     # 在仿真中同时运行 4096 个独立机器人实例（大规模并行仿真有助于高效采集数据，加速训练过程）
        num_observations = 235      # 每个机器人观测向量的维度。None：不使用固定维度，而使用后续定义的 obs_components
        use_lin_vel = True      # 是否将 线速度 作为观测向量的一部分（大规模并行仿真有助于高效采集数据，加速训练过程）
        num_privileged_obs = None   # 特权观测向量的维度。不为None：step() 会返回一个 priviledge_obs_buf （例如某些不对称训练中的 critic 观测）；None：不使用固定维度，而使用 privileged_obs_components
        num_actions = 12    # 机器人的 动作空间维度，通常对应机器人各个关节的控制输入
        env_spacing = 3.    # [m]，各仿真环境之间的空间间距，防止相互干扰。但在基于高度场（heightfields）或三角网格（trimesh）的地形中不使用
        send_timeouts = True    # 当仿真达到超时时间（例如最大仿真时长）时，将该信息反馈给训练算法，便于处理异常情况
        episode_length_s = 20   # 每个仿真 episode 的时长为 20 秒，决定了训练数据的时长及策略更新的频率

    class terrain:
        mesh_type = 'trimesh'       # 构建地形的 物理几何类型，可选 [none, plane, heightfield, trimesh]
        horizontal_scale = 0.1      # [m]，地形高度采样离散点 在水平方向的缩放因子，即每个离散点在水平方向上的实际距离
        vertical_scale = 0.005      # [m]，地形高度采样离散点 在垂直方向上高度变化的缩放因子
        border_size = 25        # [m]，地形外部的缓冲区域大小，确保机器人在靠近边界时有足够的过渡区域，防止因边界效应导致仿真不稳定
        curriculum = True       # 是否使用课程学习。True：地形的复杂度或难度会从简单逐步增加，有助于策略从易到难逐步学习应对不同场景
        # 静、动摩擦系数，对运动稳定性和滑移情况有直接影响
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.        # 弹性系数，表示碰撞时的能量损失。0 为无弹性反应
        # rough terrain only:
        measure_heights = True      # 是否在仿真中对地形的高度进行采样，这些采样离散点将作为传感器输入供机器人感知地面形态（特权信息）
        # 在机器人局部坐标系中沿 x 和 y 方向采样地形高度的离散点，1m x 1.6m 矩阵 (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 构造 x 方向上的测量点，范围 [-0.8, 0.8]，用于构建地形高度图
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # y 方向上，范围 [-0.5, 0.5]
        selected = False        # 特定地形类型
        terrain_kwargs = None   # 向特定地形传递的参数
        max_init_terrain_level = 5      # 课程学习中初始的地形难度等级，通常用于渐进式增加地形的复杂度
        # 机器人能够活动的最大范围
        terrain_length = 8.     # [m]，整个地形在 x 方向的总长度为 8米
        terrain_width = 8.      # [m]，指定地形在 y 方向的总宽度为 8米
        # 将整个地形划分为 10 * 20 个不同难度和特征的离散地形单元，提升环境复杂性和训练的鲁棒性
        num_rows = 10   # 地形单元的 难度级别
        num_cols = 20   # 地形单元的 不同类型
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]       # 不同类型地形（如平滑坡道、粗糙坡道、上楼梯、下楼梯、离散障碍等）出现的概率
        # trimesh only:
        slope_treshold = 0.75       # 斜坡角度的阈值，当坡度角 > 该阈值，将坡面校正为垂直面），避免过陡地形造成仿真计算不稳定或机器人不可行走

    class commands: # 指令
        curriculum = False      # 不采用课程学习，即生成的指令难度不会随时间增加
        max_curriculum = 1.
        num_commands = 4        # 生成的指令个数，默认包括：lin_vel_x, lin_vel_y（线速度 ）, ang_vel_yaw（偏航角速度）, heading （航向角。航向模式，会通过当前朝向误差 重新计算 偏航角速度）
        resampling_time = 10.   # [s]，每 10 秒更新一次指令，保证指令在一定时间内保持稳定
        heading_command = True  # True：会通过当前朝向误差 重新计算 偏航角速度
        class ranges:   # 各个指令的取值范围
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-1, 1]   # min max [rad/s]
            heading = [-3.14, 3.14] # 航向角 [-π, π]

    class init_state:
        pos = [0.0, 0.0, 1.]    # x,y,z [m]，机器人在仿真开始时基座的初始位置
        rot = [0.0, 0.0, 0.0, 1.0]      # x,y,z,w [quat]，初始旋转（无旋转）
        lin_vel = [0.0, 0.0, 0.0]       # x,y,z [m/s]，初始线速度
        ang_vel = [0.0, 0.0, 0.0]       # x,y,z [rad/s]，初始角速度
        default_joint_angles = { # action（动作输入）= 0.0 时，各关节的默认角度值
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P'      # 控制类型。可选 [P（位置）, V（速度）, T（力矩）]
        # 共同影响 PD控制器 的响应特性，影响关节运动的平滑性和稳定性
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]，各关节的 刚度
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]，各关节的 阻尼参数
        action_scale = 0.5      # 动作输入的缩放因子。目标角度 = action_scale * action + default_angle，有助于将策略输出限制在合理范围内
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4          # 在每个策略更新周期中，control action在仿真时步中重复使用的次数，即每 4 个仿真时步更新一次 action，从而降低策略输出的频率。（仿真频率 / 控制频率 = 4，即控制频率为250Hz）

    class asset:
        file = ""       # 机器人的模型文件（通常是 URDF 文件）
        name = "legged_robot"   # 模型名称
        foot_name = "None"      # 足端部位名称。在仿真中，足端通常用于接触检测、状态索引和接触力数据的提取
        penalize_contacts_on = []           # 与地面接触时，触发惩罚的 身体部位
        terminate_after_contacts_on = []    # 与地面接触时，触发终止仿真回合的 身体部位
        disable_gravity = False     # 是否禁用重力
        collapse_fixed_joints = True    # 是否 将由固定关节连接的刚性部分合并为一个整体，以简化计算。特殊固定关节可以通过添加标记 " <... dont_collapse="true"> 来防止合并，以保留必要的结构信息
        fix_base_link = False           # 是否固定 机器人的 base。通常在测试控制算法或进行静态分析时，可能希望固定机器人；而在动态仿真或学习任务中，通常需要机器人自由运动
        default_dof_drive_mode = 3      # 默认的 自由度（dof）驱动模式，参考 GymDofDriveModeFlags，[0（不使用任何驱动模式）, 1（基于位置控制）, 2（基于速度控制）, 3（基于effort，力矩控制）]
        self_collisions = 0     # 是否开启 自碰撞检测，[1（关闭）, 0（开启）]
        replace_cylinder_with_capsule = True    # 将 碰撞圆柱体 替换为 胶囊体。胶囊体在数值上更稳定，碰撞检测更高效，从而提升仿真速度和稳定性
        flip_visual_attachments = True          # 是否调整视觉附件的朝向，某些网格 .obj 需要从 y-up 转换到 z-up
        # 机器人各部分材料和物理特性
        density = 0.001     # 密度，影响机器人质量和惯性矩
        angular_damping = 0.    # 角阻尼系数，用以控制旋转运动中的能量耗散
        linear_damping = 0.     # 线性阻尼系数，影响机器人在直线运动中的速度衰减
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.       # 附加惯性，用于在仿真中增加关节或连杆的惯性，通常用于补偿数值误差或实现更符合实际的动力学响应
        thickness = 0.01    # 碰撞体的厚度，对于薄结构或平面结构，此参数在碰撞检测中起到决定性作用。设置合适的厚度有助于避免数值不稳定性，并确保接触检测的准确性

    class domain_rand:  # 域随机化
        randomize_friction = True   # 是否开启 随机化 摩擦系数
        friction_range = [0.5, 1.25]    # 摩擦系数的采样范围
        randomize_base_mass = False # 是否开启 随机化机器人的 base质量
        added_mass_range = [-1., 1.]    # 允许添加的质量范围
        push_robots = True  # 是否开启 定期对机器人施加外部扰动（推力），以模拟真实世界中的突发扰动
        push_interval_s = 15    # 施加间隔
        max_push_vel_xy = 1.    # 水平方向 最大推力速度
        max_push_vel_ang = 0.   # 角方向的 最大推力速度
        init_dof_pos_ratio_range = [0.5, 1.5]   # 初始关节位置比例 的随机范围
        init_base_vel_range = [-1., 1.]     # base 的初始速度 的随机范围

    class rewards:
        class scales:   # 权重
            termination = -0.0  # 仿真终止时 的惩罚
            tracking_lin_vel = 1.0  # 跟踪目标 线速度 的奖励
            tracking_ang_vel = 0.5  # # 跟踪目标 角速度 的奖励
            lin_vel_z = -2.0    # 垂直线速度 的惩罚
            ang_vel_xy = -0.05  # 水平面内角速度 的惩罚
            orientation = -0.   # 朝向（机身偏离水平面的角度）的惩罚
            torques = -0.00001  # 控制输出 力矩 的惩罚
            dof_vel = -0.       # 控制输出 关节速度 的惩罚
            dof_acc = -2.5e-7   # 控制输出 加速度 的惩罚
            base_height = -0.   # base 的高度 的惩罚
            feet_air_time =  1.0    # 足部离地时间 的奖励
            collision = -1.     # 碰撞 的惩罚
            feet_stumble = -0.0     # 足部绊跌 的惩罚
            action_rate = -0.01 # 动作变化率 的惩罚
            stand_still = -0.   # 静止状态 的惩罚

        only_positive_rewards = True    # True：当总奖励为负时，剪裁为 0，避免过早终止回合，这在训练初期有助于策略稳定性
        tracking_sigma = 0.25   # 计算跟踪奖励时误差归一化，常见于基于高斯函数的奖励设计，跟踪奖励 = exp(-error^2 / sigma)
        # 软限制值。当 关节位置、速度、扭矩超出一定比例（通常为 URDF 限制的百分比）时施加惩罚，鼓励机器人在合理范围内运动
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1. # base 的理想高度（通常低于初始高度）
        max_contact_force = 100. # 允许的最大接触力，超过该值将受到惩罚，帮助控制器保持期望的姿态与接触安全性

    class normalization:
        class obs_scales:   # 不同观测量 的缩放因子，使各项输入数值分布在相似范围内，提升神经网络训练的稳定性
            lin_vel = 2.0
            ang_vel = 0.25
            commands = [2., 2., 0.25] # 命令，与 lin_vel 和 ang_vel scales 匹配
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        # 观测值、动作值的裁剪上限，防止因异常值导致训练不稳定
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True    # 是否添加噪声
        noise_level = 1.0   # 整体噪声的强度
        class noise_scales: # 各传感器数据和状态的 噪声幅度，模拟真实环境中的不确定性和传感器误差
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:   # 可视化
        ref_env = 0 # 参考环境编号
        pos = [10, 0, 6]  # [m]，相机位置
        lookat = [11., 5, 3.]  # [m]，相机注视点
        stream_depth = False # 是否传输深度信息，for webviewer

        draw_commands = True # 在调试时是否绘制指令矢量，for debugger
        class commands:
            color = [0.1, 0.8, 0.1] # rgb，指令矢量显示的 颜色
            size = 0.5  # 指令矢量显示的 尺寸

    class sim:   # 仿真
        dt =  0.005 # 仿真时间步长，决定仿真系统的更新频率
        substeps = 1    # 每个 dt 内的子步数，用于提高数值积分精度
        gravity = [0., 0. ,-9.81]  # [m/s^2]，重力加速度方向
        up_axis = 1  # 哪个坐标轴为“向上”方向，[0（y轴）, 1（z轴）]
        no_camera = True    # 是否 禁用仿真自带的摄像头视角

        class physx:    # 针对 PhysX 物理引擎的设置
            num_threads = 10    # 并行计算时的线程数
            solver_type = 1  # 求解器类型，[0（PGS ）, 1（TGS）]
            num_position_iterations = 4 # 位置求解的 迭代次数
            num_velocity_iterations = 0 # 速度求解的 迭代次数
            contact_offset = 0.01  # [m]，碰撞检测时的 偏移距离
            rest_offset = 0.0   # [m]，解算时的 偏移
            bounce_threshold_velocity = 0.5 # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):    # 强化学习配置
    seed = 1    # 随机种子
    runner_class_name = 'OnPolicyRunner'    # 使用的训练运行器类型。 'OnPolicyRunner'：按策略梯度方法进行更新
    class policy:   # 策略网络配置
        init_noise_std = 1.0    # 初始动作噪声标准差，为策略初期提供足够探索性
        # actor 和 critic 网络的隐藏层神经元个数。较大的网络容量有助于学习复杂策略，但也可能导致训练不稳定，需要在训练中平衡模型复杂度与样本效率
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # 激活函数的类型，[elu, relu, selu, crelu, lrelu, tanh, sigmoid]。ELU 能在负值区域提供平滑性，改善训练过程中的梯度流动
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:    # 算法参数
        # training params
        value_loss_coef = 1.0   # 值函数损失在总损失中的比例，影响 critic 网络更新的权重
        use_clipped_value_loss = True   # 是否使用裁剪的价值损失，防止梯度爆炸
        clip_param = 0.2    # 策略更新裁剪参数，限制策略更新的步幅，保持新旧策略之间较小的差异
        entropy_coef = 0.01 # 熵正则化系数，用于鼓励策略探索，防止策略过早收敛到确定性行为
        # 共同决定 梯度更新的频率 与 批次大小
        num_learning_epochs = 5 # 每个策略更新周期内，对采集数据进行学习的轮数
        num_mini_batches = 4 # 每个轮次包含多个 mini-batch 更新。mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 1.e-3 # 5.e-4
        schedule = 'adaptive' # 学习率调度方式，可选 [adaptive（自适应调整，根据策略变化动态调整学习率）, fixed]
        # 共同决定 未来奖励的衰减速度与优势估计的平滑程度
        gamma = 0.99    # 折扣因子
        lam = 0.95      # GAE（广义优势估计）中的 λ 值
        desired_kl = 0.01   # 期望的 KL 散度，用于衡量新旧策略之间的差异，辅助调节策略更新步长
        max_grad_norm = 1.  # 最大梯度范数，用于梯度裁剪防止梯度爆炸
        clip_min_std = 1e-15    # 限制 action 分布最小标准差，防止策略输出过于确定性，保证一定的探索性

    class runner:
        policy_class_name = 'ActorCritic'   # 训练使用的 策略类型
        algorithm_class_name = 'PPO'    # 训练时使用的 算法类型
        num_steps_per_env = 24      # 每个环境在每次策略更新时 采集的步数，直接影响每次更新的数据量
        max_iterations = 1500       # 策略更新 的次数

        # logging
        save_interval = 50 # 每隔 50 次策略更新保存一次模型 checkpoint
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt