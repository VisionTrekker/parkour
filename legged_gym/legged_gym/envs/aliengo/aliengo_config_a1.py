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

aliengo_action_scale = 0.5
# copied from aliengo_const.h from unitree_legged_sdk
aliengo_const_dof_range = dict(
    Hip_max=1.047,  # 髋关节限制, 60 ~ -50 degree
    Hip_min=-0.873,
    Thigh_max=3.927,  # 大腿关节限制, 225 ~ -30 degree
    Thigh_min=-0.524,
    Calf_max=-0.611,  # 小腿关节限制（负值为伸展方向）, -35 ~  degree
    Calf_min=-2.775,
)

class AlienGoRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        use_lin_vel = True
        num_observations = 235  # no measure_heights makes num_obs = 48; with measure_heights makes num_obs 235

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
        default_joint_angles = {  # action = 0.0，即零动作时的目标关节角度（站立姿态）
            # 髋关节
            'FR_hip_joint': -0.1 ,  # [rad]
            'FL_hip_joint': 0.1,   # [rad]
            'RR_hip_joint': -0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            # 大腿关节
            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]
            # 小腿关节（负值表示伸展）
            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = aliengo_action_scale
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf'
        name = "aliengo"
        foot_name = "foot"  # 足部Link名称匹配模式（如"FR_foot"、"FL_foot"等包含"foot"的）
        front_hip_names = ["FL_hip_joint", "FR_hip_joint"]  # 前髋关节名称
        rear_hip_names = ["RL_hip_joint", "RR_hip_joint"]  # 后髋关节名称
        penalize_contacts_on = ["thigh", "calf"]  # 非足部区域（这里为大腿、小腿）接触地面，则触发惩罚
        terminate_after_contacts_on = ["base"]  # （机身）接触地面，则触发终止训练
        # 关节运动范围（单位：弧度）
        sdk_dof_range = aliengo_const_dof_range
        self_collisions = 1  # 1：禁用自身各部分之间的碰撞检测（提升性能）；0：启用
        dof_velocity_override = 20.  # 关节速度限制覆盖值（单位：rad/s）

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

class AlienGoRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'full'
        experiment_name = 'rough_aliengo'


#### To train the model with partial observation ####

class AliengoPlaneCfg(AlienGoRoughCfg):
    class env(AlienGoRoughCfg.env):
        use_lin_vel = False
        num_observations = 48

    class control(AlienGoRoughCfg.control):
        stiffness = {'joint': 25.}

    class domain_rand(AlienGoRoughCfg.domain_rand):
        randomize_base_mass = True

    class terrain(AlienGoRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = True

class AlienGoRoughCfgTPPO(AlienGoRoughCfgPPO):

    class algorithm(AlienGoRoughCfgPPO.algorithm):
        distillation_loss_coef = 50.

        teacher_ac_path = "logs/rough_alien/Nov08_07-55-33_full/model_1500.pt"
        teacher_policy_class_name = AlienGoRoughCfgPPO.runner.policy_class_name

        class teacher_policy(AlienGoRoughCfgPPO.policy):
            num_actor_obs = 235
            num_critic_obs = 235
            num_actions = 12

    class runner(AlienGoRoughCfgPPO.runner):
        algorithm_class_name = "TPPO"
        run_name = 'nolinvel_plane_Kp25_aclip1.5_privilegedHeights_distillation50_randomizeMass'
        experiment_name = 'teacher_aliengo'

