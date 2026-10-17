"""SO-ARM101 direct-action reinforcement-learning task.

The policy chooses incremental targets for all six actuators. There is no
scripted approach, grasp trajectory, inverse-kinematics planner or timed motion
phase. Fixed initial poses are applied only when an entire episode is reset.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw


NUM_BALLS = 6
ACTION_DIM = 6
OBSERVATION_DIM = 118
CONTROL_DT = .02
SUBSTEPS = 20
EPISODE_STEPS = 3000
REWARD_GAMMA = .995
ACTION_SCALE = np.array([.040, .035, .035, .050, .050, .060], dtype=np.float32)
INITIAL_ARM_POSE = np.array([.3879075, -.9498327, .2974578, 1.65806, 1.584, .7], dtype=np.float32)
SOURCES = np.array([[.18, -.03, .027], [.27, -.03, .027],
                    [.18, -.085, .027], [.27, -.085, .027],
                    [.18, -.14, .027], [.27, -.14, .027]], dtype=np.float32)
GOALS = np.array([[x, y, .027] for x in [.20, .26] for y in [.04, .10, .16]], dtype=np.float32)
TARGET_ORDER = np.array([5, 2, 4, 1, 3, 0], dtype=np.int64)
BALL_GOALS = GOALS[TARGET_ORDER]
THIRD_PARTY_LICENSES = """Robot model: Google DeepMind MuJoCo Menagerie,
robotstudio_so101, Apache License 2.0. The original model LICENSE is downloaded
with its assets. The task adds static containers and six physical free balls.
No scripted picking controller or trajectory code is included in this module.
"""


def _scene_xml(robot_xml):
    root = ET.parse(robot_xml).getroot()
    root.set("model", "SO101_direct_RL_six_balls")
    root.find("compiler").set("meshdir", str(robot_xml.parent / "assets"))
    root.find("option").attrib.update(timestep=".001", iterations="100",
                                      ls_iterations="50", impratio="100", tolerance="1e-6")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos=".2 -.2 1.5", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .1", rgba=".9 .92 .95 1")
    ET.SubElement(world, "geom", name="basket_floor", type="box",
                  pos=".23 -.085 .0025", size=".10 .10 .0025", rgba=".10 .40 .32 1")
    for index, (x, y, sx, sy) in enumerate([
        (.13, -.085, .003, .103), (.33, -.085, .003, .103),
        (.23, -.185, .103, .003), (.23, .015, .103, .003),
    ]):
        ET.SubElement(world, "geom", name=f"basket_wall{index}", type="box",
                      pos=f"{x} {y} .0175", size=f"{sx} {sy} .0175", rgba=".10 .40 .32 1")
    ET.SubElement(world, "geom", name="tray_floor", type="box",
                  pos=".23 .10 .0025", size=".063 .093 .0025", rgba=".77 .68 .48 1")
    for index, x in enumerate([.17, .23, .29]):
        ET.SubElement(world, "geom", name=f"tray_x_wall{index}", type="box",
                      pos=f"{x} .10 .014", size=".002 .092 .009", rgba=".64 .52 .31 1")
    for index, y in enumerate([.07, .13, .19]):
        ET.SubElement(world, "geom", name=f"tray_y_wall{index}", type="box",
                      pos=f".23 {y} .014", size=".062 .002 .009", rgba=".64 .52 .31 1")
    colors = [".95 .22 .09 1", ".95 .55 .08 1", ".92 .78 .08 1",
              ".15 .65 .35 1", ".15 .45 .90 1", ".60 .28 .85 1"]
    for index, (position, color) in enumerate(zip(SOURCES, colors)):
        body = ET.SubElement(world, "body", name=f"ball{index}",
                             pos=" ".join(str(float(v)) for v in position))
        ET.SubElement(body, "freejoint", name=f"ball_free{index}")
        ET.SubElement(body, "geom", name=f"ball{index}", type="sphere", size=".022",
                      mass=".025", friction="1.5 .02 .002", condim="6", rgba=color)
    return ET.tostring(root, encoding="unicode")


class SceneBundle:
    control_dt = CONTROL_DT
    observation_dim = OBSERVATION_DIM
    action_dim = ACTION_DIM
    num_balls = NUM_BALLS

    def __init__(self, asset_dir, max_episode_steps=EPISODE_STEPS):
        self.asset_dir = Path(asset_dir).resolve()
        self.episode_steps = int(max_episode_steps)
        if self.episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        robot_xml = self.asset_dir / "robotstudio_so101" / "so101.xml"
        self.xml = _scene_xml(robot_xml)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.lower, self.upper = self.model.actuator_ctrlrange.T.copy()
        self.initial_pose = np.clip(INITIAL_ARM_POSE, self.lower, self.upper).astype(np.float32)
        self.ball_body_ids = np.array([self.model.body(f"ball{k}").id for k in range(NUM_BALLS)])
        self.ball_geom_ids = np.array([self.model.geom(f"ball{k}").id for k in range(NUM_BALLS)])
        joints = [self.model.joint(f"ball_free{k}").id for k in range(NUM_BALLS)]
        self.ball_qpos_addresses = self.model.jnt_qposadr[joints].astype(int).copy()
        self.ball_dof_addresses = self.model.jnt_dofadr[joints].astype(int).copy()
        self.ball_xyz_indices = self.ball_qpos_addresses[:, None] + np.arange(3)
        self.ball_velocity_indices = self.ball_dof_addresses[:, None] + np.arange(3)
        self.left_body = self.model.body("gripper").id
        self.right_body = self.model.body("moving_jaw_so101_v1").id
        self.tip_ids = np.array([self.model.geom(name).id for name in
                                [f"{side}_jaw_sph_tip{i}" for side in ("fixed", "moving") for i in (1, 2, 3)]])

    def aperture(self, data):
        return data.geom_xpos[self.tip_ids].mean(axis=0)


def _potential_numpy(balls, aperture, left, right, lifted):
    distance = np.linalg.norm(aperture - balls, axis=1)
    goal_distance = np.linalg.norm(balls - BALL_GOALS, axis=1)
    height = np.clip((balls[:, 2] - SOURCES[:, 2]) / .12, 0, 1)
    return (.3 * (1 - np.tanh(15 * distance)) + .4 * (left & right) + height +
            1.5 * height * np.exp(-12 * goal_distance) +
            2. * lifted * np.exp(-50 * goal_distance))


class CPUEnv:
    observation_dim = OBSERVATION_DIM
    action_dim = ACTION_DIM
    num_balls = NUM_BALLS

    def __init__(self, bundle, seed=7, jitter=.002, max_episode_steps=None):
        self.bundle, self.model = bundle, bundle.model
        self.episode_steps = int(max_episode_steps or bundle.episode_steps)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)
        self.data = mujoco.MjData(self.model)
        self.last_info = {}
        self.reset(seed)

    @property
    def active_ball_index(self):
        unfilled = np.flatnonzero(~self.filled)
        return int(unfilled[0]) if len(unfilled) else NUM_BALLS - 1

    @property
    def target_cell(self):
        return int(TARGET_ORDER[self.active_ball_index])

    @property
    def goal(self):
        return BALL_GOALS[self.active_ball_index]

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.bundle.initial_pose
        self.joint_targets = self.bundle.initial_pose.copy()
        self.data.ctrl[:] = self.joint_targets
        for address, source in zip(self.bundle.ball_qpos_addresses, SOURCES):
            position = source.copy()
            position[:2] += self.rng.uniform(-self.jitter, self.jitter, 2)
            self.data.qpos[address:address + 3] = position
        self.step_index = 0
        self.lifted = np.zeros(NUM_BALLS, dtype=bool)
        self.bilateral_seen = np.zeros(NUM_BALLS, dtype=bool)
        self.released = np.zeros(NUM_BALLS, dtype=bool)
        self.stable_steps = np.zeros(NUM_BALLS, dtype=np.int64)
        self.lift_steps = np.zeros(NUM_BALLS, dtype=np.int64)
        self.filled = np.zeros(NUM_BALLS, dtype=bool)
        self.rewarded = np.zeros(NUM_BALLS, dtype=bool)
        self.left_contacts = np.zeros(NUM_BALLS, dtype=bool)
        self.right_contacts = np.zeros(NUM_BALLS, dtype=bool)
        self.max_heights = SOURCES[:, 2].copy()
        self.total_return = 0.
        self.success = self.done = False
        self.failure = ""
        mujoco.mj_forward(self.model, self.data)
        self.last_info = self._info()
        return self.observation()

    def observation(self):
        balls = self.data.qpos[self.bundle.ball_xyz_indices]
        velocities = self.data.qvel[self.bundle.ball_velocity_indices]
        lower, upper = self.bundle.lower, self.bundle.upper
        return np.concatenate([
            2 * (self.data.qpos[:6] - lower) / (upper - lower) - 1,
            np.clip(self.data.qvel[:6] * .1, -5, 5),
            2 * (self.joint_targets - lower) / (upper - lower) - 1,
            self.bundle.aperture(self.data) * 5,
            ((balls - SOURCES) * 5).reshape(-1), np.clip(velocities, -5, 5).reshape(-1),
            ((BALL_GOALS - balls) * 5).reshape(-1),
            self.filled, np.eye(NUM_BALLS)[self.active_ball_index], self.lifted,
            self.bilateral_seen, self.released,
            np.clip(self.stable_steps / 25., 0, 1), np.clip(self.lift_steps / 10., 0, 1),
            [max(0., 1 - self.step_index / self.episode_steps)],
        ]).astype(np.float32)

    def _jaw_contacts(self):
        contacts = np.zeros((NUM_BALLS, 2), dtype=bool)
        lookup = {int(geom): k for k, geom in enumerate(self.bundle.ball_geom_ids)}
        force = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            if contact.dist > .0005:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            ball, other = lookup.get(first), second
            if ball is None:
                ball, other = lookup.get(second), first
            if ball is None:
                continue
            body = int(self.model.geom_bodyid[other])
            if body not in (self.bundle.left_body, self.bundle.right_body):
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if force[0] > .05:
                contacts[ball, 0] |= body == self.bundle.left_body
                contacts[ball, 1] |= body == self.bundle.right_body
        return contacts[:, 0], contacts[:, 1]

    def _info(self):
        balls = self.data.qpos[self.bundle.ball_xyz_indices]
        errors = np.abs(balls[:, :2] - BALL_GOALS[:, :2]).max(axis=1)
        grid = np.zeros(NUM_BALLS, dtype=bool)
        grid[TARGET_ORDER] = self.filled
        return {
            "phase": "정책이 동작 선택", "active_ball_index": self.active_ball_index,
            "target_cell": self.target_cell, "num_balls": NUM_BALLS,
            "filled_count": int(self.filled.sum()), "filled_mask": grid.tolist(),
            "filled_by_ball": self.filled.tolist(), "remaining_balls": int(NUM_BALLS - self.filled.sum()),
            "all_target_errors": errors.tolist(), "target_error": float(errors[self.active_ball_index]),
            "ball_positions": balls.tolist(), "max_heights": self.max_heights.tolist(),
            "all_lifted": self.lifted.tolist(), "all_released": self.released.tolist(),
            "all_bilateral_seen": self.bilateral_seen.tolist(),
            "stable_steps": self.stable_steps.tolist(), "success": bool(self.success),
            "return": float(self.total_return), "steps": int(self.step_index),
            "failure": self.failure, "episode_steps": self.episode_steps,
            "control": "policy_joint_delta_targets",
        }

    def step(self, raw_actions):
        if self.done:
            raise RuntimeError("Call reset() after the whole episode ends")
        raw = np.asarray(raw_actions, dtype=np.float32)
        if raw.shape != (ACTION_DIM,) or not np.isfinite(raw).all():
            raise ValueError("Six finite policy actions are required")
        active = self.active_ball_index
        before = _potential_numpy(self.data.qpos[self.bundle.ball_xyz_indices],
                                   self.bundle.aperture(self.data), self.left_contacts,
                                   self.right_contacts, self.lifted)[active]
        old_lifted = self.lifted.copy()
        action = np.tanh(raw)
        self.joint_targets = np.clip(self.joint_targets + action * ACTION_SCALE,
                                     self.bundle.lower, self.bundle.upper)
        self.data.ctrl[:] = self.joint_targets
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        balls = self.data.qpos[self.bundle.ball_xyz_indices]
        velocities = self.data.qvel[self.bundle.ball_velocity_indices]
        distance = np.linalg.norm(self.bundle.aperture(self.data) - balls, axis=1)
        speed = np.linalg.norm(velocities, axis=1)
        errors = np.abs(balls[:, :2] - BALL_GOALS[:, :2]).max(axis=1)
        left, right = self._jaw_contacts()
        self.left_contacts, self.right_contacts = left, right
        self.bilateral_seen |= left & right
        raised = (balls[:, 2] > .08) & (distance < .055) & left & right
        self.lift_steps = np.where(raised, self.lift_steps + 1, 0)
        self.lifted |= self.lift_steps >= 10
        self.released |= self.lifted & (self.data.qpos[5] > .6) & ~left & ~right & (distance > .045)
        stable = (self.lifted & self.bilateral_seen & self.released &
                  (errors < .0055) & (np.abs(balls[:, 2] - BALL_GOALS[:, 2]) < .004) &
                  (speed < .03) & ~left & ~right & (distance > .06))
        self.stable_steps = np.where(stable, self.stable_steps + 1, 0)
        self.filled = self.stable_steps >= 25
        newly_filled = self.filled & ~self.rewarded
        self.rewarded |= newly_filled
        self.success = bool(self.filled.all())
        after = _potential_numpy(balls, self.bundle.aperture(self.data), left, right, self.lifted)[active]
        reward = (3 * (REWARD_GAMMA * after - before) - .01 - .005 * np.square(action).mean() +
                  .01 * self.filled.sum() + 5 * (self.lifted & ~old_lifted).sum() +
                  20 * newly_filled.sum() + 100 * self.success)
        self.max_heights = np.maximum(self.max_heights, balls[:, 2])
        self.total_return += float(reward)
        self.step_index += 1
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        physical_failure = (not finite or (balls[:, 2] < -.02).any() or (np.abs(balls[:, :2]) > .65).any())
        self.failure = "nonfinite_or_out_of_workspace" if physical_failure else ""
        terminated = bool(self.success or physical_failure)
        truncated = bool(self.step_index >= self.episode_steps and not terminated)
        self.done = terminated or truncated
        self.last_info = self._info()
        return self.observation(), float(reward), terminated, truncated, dict(self.last_info)


@torch.inference_mode()
def evaluate_policy(bundle, policy=None, mode="learned", episodes=1, seed=2026,
                    jitter=.002, max_episode_steps=None):
    """Optional user-run evaluation; no scripted baseline or success precondition."""
    results = []
    rng = np.random.default_rng(seed)
    for episode in range(int(episodes)):
        env = CPUEnv(bundle, seed=seed + episode, jitter=jitter, max_episode_steps=max_episode_steps)
        obs = env.observation()
        for _ in range(env.episode_steps):
            if mode == "zero_action":
                action = np.zeros(ACTION_DIM)
            elif mode == "random":
                action = rng.standard_normal(ACTION_DIM)
            elif mode == "learned" and policy is not None:
                action = policy.actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).cpu().numpy()
            else:
                raise ValueError("Use learned with a policy, random, or zero_action")
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        results.append(info)
    if not results:
        raise ValueError("episodes must be at least one")
    return {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_filled_count": float(np.mean([r["filled_count"] for r in results])),
        "remaining_balls": float(np.mean([r["remaining_balls"] for r in results])),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_xy_error": float(np.mean([np.mean(r["all_target_errors"]) for r in results])),
        "lift_rate": float(np.mean([np.mean(r["all_lifted"]) for r in results])),
        "release_rate": float(np.mean([np.mean(r["all_released"]) for r in results])),
        "episodes": results, "seed": int(seed), "jitter_m": float(jitter),
        "backend": "CPU MuJoCo evaluation", "mode": mode,
        "success_contract": "all six physical balls currently stable in assigned cells after real grasp, lift and release",
    }


class WarpEnv:
    """Direct policy actions in batched MuJoCo Warp; only whole episodes reset."""
    observation_dim = OBSERVATION_DIM
    action_dim = ACTION_DIM
    num_balls = NUM_BALLS

    def __init__(self, bundle, num_envs=8, device="cuda:0", seed=7, jitter=.002,
                 capture_graph=True, max_episode_steps=None):
        self.bundle, self.num_envs = bundle, int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.episode_steps = int(max_episode_steps or bundle.episode_steps)
        self.device = torch.device(device)
        self.wp_device = str(self.device) if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.stream = (wp.stream_from_torch(torch.cuda.current_stream(self.device))
                       if self.device.type == "cuda" else None)
        self.jitter, self.graph = float(jitter), None
        with self.scope():
            self.model = mjw.put_model(bundle.model)
            self.data = mjw.make_data(bundle.model, nworld=self.num_envs, nconmax=192, njmax=600)
            self.qpos, self.qvel, self.ctrl = [wp.to_torch(getattr(self.data, key)) for key in ("qpos", "qvel", "ctrl")]
            self.geom_xpos = wp.to_torch(self.data.geom_xpos)
            self.sources = torch.as_tensor(SOURCES, dtype=torch.float32, device=self.device)
            self.goals = torch.as_tensor(BALL_GOALS, dtype=torch.float32, device=self.device)
            self.scale = torch.as_tensor(ACTION_SCALE, device=self.device)
            self.lower, self.upper = [torch.as_tensor(v, dtype=torch.float32, device=self.device)
                                      for v in (bundle.lower, bundle.upper)]
            self.initial_pose = torch.as_tensor(bundle.initial_pose, device=self.device)
            self.ball_xyz_indices = torch.as_tensor(bundle.ball_xyz_indices, dtype=torch.long, device=self.device)
            self.ball_velocity_indices = torch.as_tensor(bundle.ball_velocity_indices, dtype=torch.long, device=self.device)
            self.ball_indices = torch.arange(NUM_BALLS, device=self.device)
            self.world = torch.arange(self.num_envs, device=self.device)
            self.grid_order = torch.as_tensor(TARGET_ORDER, dtype=torch.long, device=self.device)
            self.geom_bodyid = torch.as_tensor(bundle.model.geom_bodyid, dtype=torch.long, device=self.device)
            self.geom_to_ball = torch.full((bundle.model.ngeom,), -1, dtype=torch.long, device=self.device)
            self.geom_to_ball[torch.as_tensor(bundle.ball_geom_ids, dtype=torch.long, device=self.device)] = self.ball_indices
            self.contact_ids = wp.array(np.arange(self.data.naconmax, dtype=np.int32), dtype=wp.int32, device=self.wp_device)
            self.contact_force_buf = wp.zeros(self.data.naconmax, dtype=wp.spatial_vector, device=self.wp_device)
            self.contact_force_view = wp.to_torch(self.contact_force_buf)
            self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.joint_targets = self.initial_pose.expand(self.num_envs, ACTION_DIM).clone()
            self.returns = torch.zeros(self.num_envs, device=self.device)
            shape = (self.num_envs, NUM_BALLS)
            self.lift_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
            self.stable_steps = torch.zeros_like(self.lift_steps)
            self.lifted = torch.zeros(shape, dtype=torch.bool, device=self.device)
            self.bilateral_seen = torch.zeros_like(self.lifted)
            self.released = torch.zeros_like(self.lifted)
            self.filled = torch.zeros_like(self.lifted)
            self.rewarded = torch.zeros_like(self.lifted)
            self.left_contacts = torch.zeros_like(self.lifted)
            self.right_contacts = torch.zeros_like(self.lifted)
            self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.reset_mask = torch.ones_like(self.success)
            self.wp_reset_mask = wp.from_torch(self.reset_mask)
            initial_rng_state = self.generator.get_state()
            self._reset(self.reset_mask)
            mjw.forward(self.model, self.data)
            if self.device.type == "cuda" and capture_graph:
                for _ in range(SUBSTEPS):
                    mjw.step(self.model, self.data)
                wp.synchronize_device(self.wp_device)
                with wp.ScopedCapture(device=self.wp_device) as capture:
                    for _ in range(SUBSTEPS):
                        mjw.step(self.model, self.data)
                self.graph = capture.graph
                self.generator.set_state(initial_rng_state)
                self._reset(torch.ones_like(self.success))
                mjw.forward(self.model, self.data)

    def scope(self):
        stack = contextlib.ExitStack()
        stack.enter_context(wp.ScopedDevice(self.wp_device))
        if self.stream is not None:
            stack.enter_context(wp.ScopedStream(self.stream))
        return stack

    def _reset(self, mask):
        self.reset_mask.copy_(mask)
        mjw.reset_data(self.model, self.data, reset=self.wp_reset_mask)
        arm = self.initial_pose.expand(self.num_envs, ACTION_DIM)
        self.qpos[:, :6].copy_(torch.where(mask[:, None], arm, self.qpos[:, :6]))
        self.joint_targets.copy_(torch.where(mask[:, None], arm, self.joint_targets))
        self.ctrl.copy_(torch.where(mask[:, None], arm, self.ctrl))
        positions = self.sources.expand(self.num_envs, NUM_BALLS, 3).clone()
        positions[:, :, :2] += self.jitter * (2 * torch.rand(
            (self.num_envs, NUM_BALLS, 2), device=self.device, generator=self.generator) - 1)
        self.qpos[:, self.ball_xyz_indices] = torch.where(mask[:, None, None], positions,
                                                         self.qpos[:, self.ball_xyz_indices])
        self.steps.masked_fill_(mask, 0)
        self.returns.masked_fill_(mask, 0)
        for field in (self.lift_steps, self.stable_steps):
            field.masked_fill_(mask[:, None], 0)
        for field in (self.lifted, self.bilateral_seen, self.released, self.filled,
                      self.rewarded, self.left_contacts, self.right_contacts):
            field.masked_fill_(mask[:, None], False)
        self.success.masked_fill_(mask, False)

    def _eef(self):
        return self.geom_xpos[:, self.bundle.tip_ids].mean(dim=1)

    def _balls(self):
        return self.qpos[:, self.ball_xyz_indices]

    def _velocities(self):
        return self.qvel[:, self.ball_velocity_indices]

    def _active(self):
        # Goal bookkeeping changes no arm or ball coordinate.
        active = (~self.filled).to(torch.int64).argmax(dim=1)
        return torch.where(self.filled.all(dim=1), NUM_BALLS - 1, active)

    def observation(self):
        balls, velocities = self._balls(), self._velocities()
        return torch.cat([
            2 * (self.qpos[:, :6] - self.lower) / (self.upper - self.lower) - 1,
            (self.qvel[:, :6] * .1).clamp(-5, 5),
            2 * (self.joint_targets - self.lower) / (self.upper - self.lower) - 1,
            self._eef() * 5,
            ((balls - self.sources) * 5).reshape(self.num_envs, -1),
            velocities.clamp(-5, 5).reshape(self.num_envs, -1),
            ((self.goals - balls) * 5).reshape(self.num_envs, -1),
            self.filled.float(), torch.nn.functional.one_hot(self._active(), NUM_BALLS).float(),
            self.lifted.float(), self.bilateral_seen.float(), self.released.float(),
            (self.stable_steps / 25.).clamp(0, 1), (self.lift_steps / 10.).clamp(0, 1),
            (1 - self.steps[:, None] / self.episode_steps).clamp_min(0),
        ], dim=1)

    def _potential(self):
        balls = self._balls()
        distance = (self._eef()[:, None, :] - balls).norm(dim=2)
        goal_distance = (balls - self.goals).norm(dim=2)
        height = ((balls[:, :, 2] - self.sources[:, 2]) / .12).clamp(0, 1)
        return (.3 * (1 - torch.tanh(15 * distance)) + .4 * (self.left_contacts & self.right_contacts) +
                height + 1.5 * height * torch.exp(-12 * goal_distance) +
                2 * self.lifted * torch.exp(-50 * goal_distance))

    def _jaw_contacts(self):
        mjw.contact_force(self.model, self.data, self.contact_ids, False, self.contact_force_buf)
        contact = self.data.contact
        geoms = wp.to_torch(contact.geom).long()
        world = wp.to_torch(contact.worldid).long()
        active = torch.arange(geoms.shape[0], device=self.device) < wp.to_torch(self.data.nacon)[0]
        valid_geom = (geoms >= 0).all(dim=1) & (geoms < self.bundle.model.ngeom).all(dim=1)
        safe = geoms.clamp(0, self.bundle.model.ngeom - 1)
        first, second = self.geom_to_ball[safe[:, 0]], self.geom_to_ball[safe[:, 1]]
        ball = torch.where(first >= 0, first, second)
        other = torch.where(first >= 0, safe[:, 1], safe[:, 0])
        valid = (active & valid_geom & (ball >= 0) & (world >= 0) & (world < self.num_envs) &
                 (wp.to_torch(contact.dist) <= .0005) & (self.contact_force_view[:, 0] > .05))
        bodies = self.geom_bodyid[other]
        index = world.clamp(0, self.num_envs - 1) * NUM_BALLS + ball.clamp(0, NUM_BALLS - 1)
        result = []
        for body in (self.bundle.left_body, self.bundle.right_body):
            out = torch.zeros(self.num_envs * NUM_BALLS, device=self.device)
            out.scatter_add_(0, index, (valid & (bodies == body)).float())
            result.append(out.reshape(self.num_envs, NUM_BALLS) > 0)
        return result

    @torch.no_grad()
    def step(self, raw_actions):
        with self.scope():
            raw = torch.as_tensor(raw_actions, dtype=torch.float32, device=self.device)
            if raw.shape != (self.num_envs, ACTION_DIM) or not bool(torch.isfinite(raw).all()):
                raise ValueError("Actions must be finite with shape (num_envs, 6)")
            active = self._active()
            before = self._potential()[self.world, active]
            old_lifted = self.lifted.clone()
            action = raw.tanh()
            self.joint_targets.copy_((self.joint_targets + action * self.scale).clamp(self.lower, self.upper))
            self.ctrl.copy_(self.joint_targets)
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                for _ in range(SUBSTEPS):
                    mjw.step(self.model, self.data)
            mjw.forward(self.model, self.data)
            balls, velocities = self._balls(), self._velocities()
            distance = (self._eef()[:, None, :] - balls).norm(dim=2)
            speed = velocities.norm(dim=2)
            errors = (balls[:, :, :2] - self.goals[:, :2]).abs().amax(dim=2)
            left, right = self._jaw_contacts()
            self.left_contacts.copy_(left)
            self.right_contacts.copy_(right)
            self.bilateral_seen.logical_or_(left & right)
            raised = (balls[:, :, 2] > .08) & (distance < .055) & left & right
            self.lift_steps.copy_(torch.where(raised, self.lift_steps + 1, 0))
            self.lifted.logical_or_(self.lift_steps >= 10)
            self.released.logical_or_(self.lifted & (self.qpos[:, 5:6] > .6) &
                                      ~left & ~right & (distance > .045))
            stable = (self.lifted & self.bilateral_seen & self.released &
                      (errors < .0055) & ((balls[:, :, 2] - self.goals[:, 2]).abs() < .004) &
                      (speed < .03) & ~left & ~right & (distance > .06))
            self.stable_steps.copy_(torch.where(stable, self.stable_steps + 1, 0))
            self.filled.copy_(self.stable_steps >= 25)
            newly_filled = self.filled & ~self.rewarded
            self.rewarded.logical_or_(newly_filled)
            self.success.copy_(self.filled.all(dim=1))
            after = self._potential()[self.world, active]
            reward = (3 * (REWARD_GAMMA * after - before) - .01 - .005 * action.square().mean(dim=1) +
                      .01 * self.filled.sum(dim=1) + 5 * (self.lifted & ~old_lifted).sum(dim=1) +
                      20 * newly_filled.sum(dim=1) + 100 * self.success)
            self.returns.add_(reward)
            self.steps.add_(1)
            finite = torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1)
            physical_failure = (~finite | (balls[:, :, 2] < -.02).any(dim=1) |
                                (balls[:, :, :2].abs() > .65).any(dim=2).any(dim=1))
            terminated = physical_failure | self.success
            truncated = (self.steps >= self.episode_steps) & ~terminated
            done = terminated | truncated
            final_obs = torch.nan_to_num(self.observation(), nan=0., posinf=0., neginf=0.)
            filled_count = self.filled.sum(dim=1)
            grid = torch.zeros_like(self.filled)
            grid[:, self.grid_order] = self.filled
            info = {
                "final_obs": final_obs, "episode_returns": self.returns[done].clone(),
                "episode_success": self.success[done].clone(),
                "episode_filled_count": filled_count[done].clone(),
                "episode_xy_error": errors[done].amax(dim=1).clone(),
                "filled_count": filled_count.clone(), "filled_mask": grid.clone(),
                "active_ball_index": active.clone(), "target_cell": self.grid_order[active],
            }
            if bool(done.any()):
                self._reset(done)
                mjw.forward(self.model, self.data)
            return self.observation(), torch.nan_to_num(reward), terminated, truncated, info
