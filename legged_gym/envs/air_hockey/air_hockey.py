import os

import numpy as np
import torch
import logging
from pytorch3d.transforms import *
from air_hockey_challenge.environments.planar.base import AirHockeyBase as MCJBase
from air_hockey_challenge.environments.position_control_wrapper import PositionControlPlanar
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot

from .air_hockey_config import AirHockeyCfg
from .transform import T_mat_from_body_state


class AirHockeyBase(LeggedRobot):
    def __init__(self, cfg: AirHockeyCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.num_ctrl = len(cfg.control.control_joint_idx) * 2

        self.puck_roll = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.puck_pitch = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.step_count = 0

    def clone_mujoco_controller(self, controller: PositionControlPlanar):
        # clone the mujoco controller and delete the original to reduce memory usage
        self.base_interp_traj = controller.interpolate_trajectory
        self.base_enforce_safety_limits = controller.enforce_safety_limits
        self.ctrl_dt = self.dt * self.cfg.control.decimation
        self.interp_order = controller.interp_order
        self.prev_pos = controller.prev_pos
        self.prev_vel = controller.prev_vel
        self.prev_acc = controller.prev_acc
        self._num_env_joints = controller._num_env_joints
        self._num_coeffs = controller._num_coeffs
        self._timestep = self.dt
        self._n_intermediate_steps = self.cfg.control.decimation
        self.jerk = controller.jerk
        self.get_reset_puck_pos = controller.get_reset_puck_pos
        self.hit_range = controller.hit_range

    def puck_2d_in_robot_frame(self, puck_actor, T_base_actor, type):
        puck_base = torch.zeros_like(puck_actor)
        if type == 'pose':
            ea_actor_puck = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
            ea_actor_puck[:, 2] = puck_actor[:, 2]
            r_actor_puck = euler_angles_to_matrix(ea_actor_puck, convention='XYZ')
            # q_actor_puck = matrix_to_quaternion(r_actor_puck)
            # q_base_puck = quaternion_multiply(q_base_actor, q_actor_puck)
            #
            # r_base_puck = quaternion_to_matrix(q_base_puck)
            #
            # r_actor_base = quaternion_to_matrix(q_actor_base)
            #
            # T_actor_base = torch.eye(4, device=self.device, dtype=torch.float)
            # T_actor_base[:3, :3] = r_actor_base
            # T_actor_base[:3, 3] = t_actor_base
            #
            # T_base_actor = T_actor_base.inverse()

            T_actor_puck = torch.eye(4, device=self.device, dtype=torch.float).unsqueeze(0).repeat(self.num_envs, 1, 1)
            T_actor_puck[..., :3, :3] = r_actor_puck
            T_actor_puck[..., :3, 3] = puck_actor[:, :3]

            T_base_puck = torch.matmul(T_base_actor, T_actor_puck)

            puck_base[:, 0] = T_base_puck[:, 0, 3]
            puck_base[:, 1] = T_base_puck[:, 1, 3]
            # get yaw angle
            ea_base_puck = matrix_to_euler_angles(T_base_puck[..., :3, :3], convention='XYZ')
            puck_base[:, 2] = ea_base_puck[..., 2]
        else:
            vel_lin = torch.cat(
                (puck_actor[:, :2], torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)),
                dim=1)
            vel_ang = torch.cat(
                [torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float), puck_actor[:, 2:3]],
                dim=1)

            rot_base_actor = T_base_actor[:, :3, :3]

            vel_lin_base = torch.matmul(rot_base_actor, vel_lin.unsqueeze(-1)).squeeze(-1)
            vel_ang_base = torch.matmul(rot_base_actor, vel_ang.unsqueeze(-1)).squeeze(-1)

            puck_base[:, :2] = vel_lin_base[:, :2]
            puck_base[:, 2] = vel_ang_base[:, 2]

        return puck_base

    def interpolate_trajectory(self, action):
        action = action.reshape((2, self._num_env_joints))
        traj, self.prev_pos, self.prev_vel, self.prev_acc, self.jerk = self.base_interp_traj(self.ctrl_dt,
                                                                                             self.interp_order,
                                                                                             self.prev_pos,
                                                                                             self.prev_vel,
                                                                                             self.prev_acc,
                                                                                             self._num_env_joints,
                                                                                             self._num_coeffs,
                                                                                             self._timestep,
                                                                                             self._n_intermediate_steps,
                                                                                             action)
        return traj

    def enforce_safety_limits(self, desired_pos, desired_vel):
        return np.ones((self.num_envs, self._num_env_joints)), np.ones((self.num_envs, self._num_env_joints))

    # action shape: (num_envs, 2: [q, dq], num_joints=3)
    def step(self, actions):
        # make a dummy step for debug

        # TODO: adjust the action dimension and its cfg
        # clip_actions = self.cfg.normalization.clip_actions
        # self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # actions to tensor
        traj = None
        if actions is not None:
            actions = actions.detach().cpu().numpy()

            traj = np.apply_along_axis(self.interpolate_trajectory, 1, actions)

        self.render()

        # TODO: translate the action
        print('step: ', self.step_count)
        self.step_count += 1

        for _ in range(self.cfg.control.decimation):
            # check if any of self.T_base_actor is nan
            # only start control when the T_base_actor is valid
            if not torch.isnan(self.T_base_actor).any():
                actions_step = np.array([next(t) for t in traj])  # num_envs x [q qd qdd] x num_joints
                self.ctrl_actions = torch.from_numpy(actions_step).to(torch.float32).to(self.device)
                # zero actions
                # self.ctrl_actions = torch.zeros(self.num_envs, 3, self._num_env_joints).to(self.device)
                self.ctrl_torques = self._compute_torques(self.ctrl_actions).view(self.ctrl_torques.shape)
                # TODO: how to set the torques
                self.torques[:, self.ctrl_joints_idx[0]] = self.ctrl_torques[:, 0]
                self.torques[:, self.ctrl_joints_idx[1]] = self.ctrl_torques[:, 1]
                self.torques[:, self.ctrl_joints_idx[2]] = self.ctrl_torques[:, 2]
                # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            else:
                env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
                self.reset_idx(env_ids)
                self._init_buffers()

                self.gym.simulate(self.sim)
                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
                self.post_physics_step()

                # return clipped obs, clipped states (None), rewards, dones and infos
                # clip_obs = self.cfg.normalization.clip_observations
                # self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # if self.privileged_obs_buf is not None:
        #     self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions: torch.Tensor):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        # TODO: enforce safety limits
        clipped_pos = actions.view(self.num_envs, 3, self._num_env_joints)[..., 0, ...]
        clipped_vel = actions.view(self.num_envs, 3, self._num_env_joints)[..., 1, ...]
        clipped_acc = actions.view(self.num_envs, 3, self._num_env_joints)[..., 2, ...]

        error = clipped_pos - self.ctrl_dof_pos

        i_error = 0  # Ki=0 anyways
        torques = self.ctrl_p_gains * error + self.ctrl_d_gains * (clipped_vel - self.ctrl_dof_vel) + i_error

        return torch.clip(torques, -self.ctrl_torque_limits, self.ctrl_torque_limits)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _post_physics_step_callback(self):
        pass

    def compute_observations(self):
        # self.env_info['puck_pos_ids'] = [0, 1, 2]
        # self.env_info['puck_vel_ids'] = [3, 4, 5]
        # self.env_info['joint_pos_ids'] = [6, 7, 8]
        # self.env_info['joint_vel_ids'] = [9, 10, 11]

        puck_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, self.puck_pos_idx, 0]
        puck_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, self.puck_pos_idx, 1]
        joint_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, self.ctrl_joints_idx, 0]
        joint_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, self.ctrl_joints_idx, 1]

        puck_pos = self.puck_2d_in_robot_frame(puck_pos, self.T_base_actor, type='pose')
        puck_vel = self.puck_2d_in_robot_frame(puck_vel, self.T_base_actor, type='vel')

        self.obs_buf = torch.cat((puck_pos, puck_vel, joint_pos, joint_vel), dim=1)

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for name in self.cfg.control.control_joint_idx.keys():
            i = self.dof_names.index(name)
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.ctrl_torques = torch.zeros(self.num_envs, len(self.ctrl_joints_idx), dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.ctrl_dof_pos = self.dof_pos[..., self.ctrl_joints_idx]
        self.ctrl_dof_vel = self.dof_vel[..., self.ctrl_joints_idx]
        self.ctrl_actions = torch.Tensor(self.num_envs, 3, len(self.ctrl_joints_idx)).to(
            self.device)  # 3: q, qd, qdd
        self.ctrl_p_gains = self.p_gains[self.ctrl_joints_idx]
        self.ctrl_d_gains = self.p_gains[self.ctrl_joints_idx]
        self.ctrl_torque_limits = self.torque_limits[self.ctrl_joints_idx]

        # body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # self.body_state = gymtorch.wrap_tensor(body_state)
        # self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, -1)[..., :3]
        # self.body_quat = self.body_state.view(self.num_envs, self.num_bodies, -1)[..., 3:7]
        #
        # self.body_vel = self.body_state.view(self.num_envs, self.num_bodies, -1)[..., 7:10]
        # self.body_ang_vel = self.body_state.view(self.num_envs, self.num_bodies, -1)[..., 10:13]
        # self.robot_base_body_id = self.body_names.index(self.cfg.control.robot_base_body)
        #
        # # create a self.num_envs * 4 *4 eye matrix
        # self.T_world_base = torch.eye(4, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(
        #     0).repeat(self.num_envs, 1, 1)
        #
        # self.T_world_actor = torch.eye(4, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(
        #     0).repeat(self.num_envs, 1, 1)
        # self.T_world_actor[..., :3, 3] = self.base_pos
        # self.T_world_actor[..., :3, :3] = quaternion_to_matrix(self.base_quat)
        #
        # self.T_world_base[..., :2, 3] = self.body_pos[:, self.robot_base_body_id, :2]
        # print(self.body_pos[0, self.robot_base_body_id, :2])
        # print(self.body_pos[0, self.robot_base_body_id, :])
        # self.T_world_base[..., 2, 3] = 0
        # r_actor_base = quaternion_to_matrix(self.body_quat[:, self.robot_base_body_id, :])
        # self.T_world_base[..., :3, :3] = r_actor_base
        #
        # self.T_base_actor = torch.matmul(torch.inverse(self.T_world_base), self.T_world_actor)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        idx = self.cfg.control.control_joint_idx.keys()
        self.ctrl_joints_idx = np.zeros(len(idx), dtype=np.int32);
        dof_dict = self.gym.get_asset_dof_dict(robot_asset)
        for name in idx:
            self.ctrl_joints_idx[self.cfg.control.control_joint_idx[name]] = int(dof_dict[name])
        all_numbers = np.arange(self.num_actions)
        free_joints = np.setdiff1d(all_numbers, self.ctrl_joints_idx)
        self.ctrl_joints_idx_full = np.concatenate((self.ctrl_joints_idx, free_joints))
        self.ctrl_joints_idx_tensor = torch.from_numpy(self.ctrl_joints_idx_full).to(torch.int32).to(self.device)

        self.num_bodies = len(body_names)
        self.body_names = body_names
        self.num_dofs = len(self.dof_names)

        # get the dof indices of the obs
        self.puck_pos_idx = np.zeros(len(self.cfg.init_state.puck_pos), dtype=np.int32)
        for name in self.cfg.init_state.puck_pos.keys():
            self.puck_pos_idx[self.cfg.init_state.puck_pos[name]] = int(dof_dict[name])

        self.puck_vel_idx = np.zeros(len(self.cfg.init_state.puck_vel), dtype=np.int32)
        for name in self.cfg.init_state.puck_vel.keys():
            self.puck_vel_idx[self.cfg.init_state.puck_vel[name]] = int(dof_dict[name])

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            # body_names = self._process_rigid_body_props(body_props, i)
            body_props = self._process_rigid_body_props(body_props, i, self.body_names)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            # set body colors
            for body_name in self.cfg.asset.colors.keys():
                body_idx = body_names.index(body_name)
                self.gym.set_rigid_body_color(env_handle, actor_handle, body_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(*self.cfg.asset.colors[body_name]))

                self.envs.append(env_handle)
                self.actor_handles.append(actor_handle)

                ctrl_actor_idx = np.zeros(len(self.cfg.control.control_joint_idx.keys()), dtype=np.int32)
                # iterate all actuator joint names
                for i in range(self.gym.get_asset_actuator_count(robot_asset)):
                    joint_name = self.gym.get_asset_actuator_joint_name(robot_asset, i)
                if joint_name in self.cfg.control.control_joint_idx.keys():
                    ctrl_actor_idx[self.cfg.control.control_joint_idx[joint_name]] = i
                self.ctrl_actor_idx = torch.from_numpy(ctrl_actor_idx).to(torch.int32).to(self.device)

    def _process_rigid_shape_props(self, rigid_shape_props_asset, env_id):
        # TODO read friction
        return rigid_shape_props_asset

    def _process_dof_props(self, dof_props_asset, env_id):
        # TODO read pos/vel/torque limits
        return super(AirHockeyBase, self)._process_dof_props(dof_props_asset, env_id)

    def _process_rigid_body_props(self, body_props, env_id, names):
        for prop, name in zip(body_props, names):
            # if name starts with table
            if name.startswith('table'):
                prop.mass = 5000
        # return super(AirHockeyBase, self)._process_rigid_body_props(body_props, env_id)
        return body_props

    def _post_physics_step_callback(self):
        pass

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # this commented line is used to randomize the initial position of the robot
        self.dof_pos[
            env_ids] = self.default_dof_pos  # * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
        # reset puck pos
        # reset_puck_pos = self.get_reset_puck_pos(self.hit_range)
        # self.dof_pos[env_ids, self.puck_pos_idx[0]] = reset_puck_pos[0]
        # self.dof_pos[env_ids, self.puck_pos_idx[1]] = reset_puck_pos[1]
        # self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset(self):
        # TODO: reset prev_pos, prev_vel, prev_acc, self.prev_controller_cmd_pos
        # we have 10 joints, but only control 3 of them, so num_actions for the controller is 3, but 10 for the env
        # override reset to work around this
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_ctrl, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf.zero_()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
        """
        # self.rewards
        pass
