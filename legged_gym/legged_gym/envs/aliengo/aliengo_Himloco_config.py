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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AlienGoRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096 # 并行仿真的环境数量（需根据GPU显存调整）
        num_one_step_observations = 45  # 单步观测维度（原始传感器数据）
        num_observations = num_one_step_observations * 6    # 最终观测维度（含6步历史）
        num_one_step_privileged_obs = 45 + 3 + 3 + 187 # 单部特权观测 维度，包含外部力（3维力+3维力矩）和地形扫描（187个点）
        num_privileged_obs = num_one_step_privileged_obs * 1 # 特权观测 最终维度 if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12    # 动作空间维度（12个关节）
        env_spacing = 3.  # # 环境之间的间距（单位：米），not used with heightfields/trimeshes
        send_timeouts = True # 是否发送超时信号给算法，send time out information to the algorithm
        episode_length_s = 20 # 单次训练Episode的时长（秒），episode length in seconds

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.55] #  初始位置（x,y,z）单位：米
        default_joint_angles = { # action = 0.0，即零动作时的目标关节角度（站立姿态）
            # 髋关节
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]
            # 大腿关节
            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            # 小腿关节（负值表示伸展）
            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'  # 控制类型（P=位置控制，T=力矩控制）
        stiffness = {'joint': 40.0}  # 关节刚度（单位：N·m/rad）
        damping = {'joint': 2.0}     # 关节阻尼（单位：N·m·s/rad）
        action_scale = 0.5  # 动作缩放因子（目标角度 = 动作 * scale + 默认角度）
        decimation = 4  # 动作控制频率（仿真频率 / 控制频率 = 4，即控制频率为250Hz，应与真实机器人控制频率对齐）
        hip_reduction = 1.0 # 髋关节扭矩缩放因子（用于平衡前后腿负载）

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf'
        name = "aliengo"    # 机器人标识名称
        foot_name = "foot"  # 足部Link名称匹配模式（如"FR_foot"、"FL_foot"等包含"foot"的）
        penalize_contacts_on = ["thigh", "calf", "base"]    # 非足部区域（这里为大腿、小腿）接触地面，则触发惩罚
        terminate_after_contacts_on = ["base"]      # （机身）接触地面，则触发终止训练
        privileged_contacts_on = ["base", "thigh", "calf"]  # 特权接触检测区域
        self_collisions = 1 # 1：禁用自身各部分之间的碰撞检测（提升性能）；0：启用
        flip_visual_attachments = True # 翻转视觉模型坐标系（Y-up转Z-up），许多 .obj meshes 必须从 y-up 转到 z-up
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -0.0  # Episode终止惩罚：未启用。设为负值（如-10.0）可在跌倒时给予额外惩罚
            tracking_lin_vel = 1.0  # 线速度跟踪 奖励：机器人实际线速度与指令速度的匹配度。权重最大，主导前进/后退训练
            tracking_ang_vel = 0.5  # 角速度跟踪 奖励：控制转向精度。若机器人转弯不稳定，可降低权重（如0.3）
            lin_vel_z = -2.0    # 垂直速度 惩罚：防止机身跳跃。若机器人跳跃频繁，增大惩罚（如-5.0）
            ang_vel_xy = -0.05  # 俯仰/横滚角速度 惩罚：抑制机身倾斜。跌倒时增大（如-0.2）
            orientation = -0.2  # 姿态 惩罚：机身偏离水平面的角度惩罚。地面不平时可减小（如-0.1）
            dof_acc = -2.5e-7   # 关节加速度 惩罚：抑制关节突变运动。若步态抖动，增大惩罚（如-1e-6）
            joint_power = -2e-5 # 关节功率 惩罚：降低能耗。需平衡运动效率，过高惩罚会导致动作迟缓
            base_height = -1.0  # 机身高度 惩罚：当高度低于 base_height_target (0.3m) 时触发。增大惩罚（如-2.0）可强化贴地
            foot_clearance = -0.01  # 足部离地高度惩罚：防止抬脚过高。值越负（如-0.1）越限制抬腿幅度
            action_rate = -0.01 # 动作变化率惩罚：相邻动作差值惩罚。调大（如-0.05）可使运动更连续
            smoothness = -0.01  # 平滑性惩罚：高阶动作导数惩罚。复杂地形中可适当降低
            feet_air_time =  0.0    # 足部空中时间奖励：当前未启用，设为正数可鼓励迈步（如0.2）
            collision = -0.0    # 剧烈碰撞惩罚：未启用。检测超过 max_contact_force (100N) 的接触，设为负值（如-0.1）可防硬件过载
            feet_stumble = -0.0 # 足部打滑惩罚：检测足部横向滑动。若打滑严重，设为负值（如-0.1
            stand_still = -0.
            torques = -0.0  # 关节力矩惩罚：未启用。若仿真关节过热，设为负值（如-1e-4）
            dof_vel = -0.0  # 关节速度惩罚：抑制高速抖动。若关节振荡，设为负值（如-0.01）
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0

        only_positive_rewards = False # 负奖励保留：为True时总奖励不低于零，避免早期训练频繁终止。复杂任务建议保持False
        tracking_sigma = 0.25   # 跟踪奖励的高斯分布标准差 = exp(-error^2 / sigma)
        soft_dof_pos_limit = 0.95   # 关节位置软限位：关节角度超过URDF限位95%时触发惩罚。调低（如0.9）可提前约束
        soft_dof_vel_limit = 0.95   # 关节速度软限位：超过最大速度95%时惩罚。保护电机模型不过载
        soft_torque_limit = 0.95    # 关节力矩软限位：超过额定扭矩95%时惩罚。防止仿真数值发散
        base_height_target = 0.30   # 机身目标高度（低于初始高度0.55m）
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.20 # 足部离地高度目标：负值表示允许触地。调为正值可强制抬腿（如0.05m）

class AlienGoRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01 # 熵系数（鼓励探索）
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_aliengo'

  