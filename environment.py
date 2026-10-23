"""SO-ARM101 direct-action reinforcement-learning task.

The policy chooses incremental targets for all six actuators. There is no
scripted approach, grasp trajectory, inverse-kinematics planner or timed motion
phase. Fixed initial poses are applied only when an entire episode is reset.
Both actor and critic observe camera RGB and robot proprioception only; exact
object positions and contact flags remain internal to reward and diagnostics.
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


TASK_ID = "so101_one_ball_flat_plate_vision_v1"
NUM_BALLS = 1
ACTION_DIM = 6
OBSERVATION_DIM = 25
CAMERA_NAMES = ("wrist_cam", "overview_cam")
IMAGE_SIZE = 64
IMAGE_INTERVAL = 4
FRAME_STACK = 2
IMAGE_SHAPE = (FRAME_STACK * len(CAMERA_NAMES) * 3, IMAGE_SIZE, IMAGE_SIZE)
OBSERVATION_SPEC = {
    "mode": "rgb_and_proprioception",
    "image_shape": list(IMAGE_SHAPE),
    "image_dtype": "uint8",
    "camera_names": list(CAMERA_NAMES),
    "frame_stack": FRAME_STACK,
    "frame_order": "oldest_to_latest_then_camera_then_rgb",
    "image_interval_control_steps": IMAGE_INTERVAL,
    "proprio_dim": OBSERVATION_DIM,
    "proprio_fields": ["joint_positions", "joint_velocities", "joint_targets",
                      "previous_action", "episode_time_remaining"],
}
CONTROL_DT = .02
SUBSTEPS = 20
EPISODE_STEPS = 1000
REWARD_GAMMA = .995
ACTION_SCALE = np.array([.040, .035, .035, .050, .050, .060], dtype=np.float32)
# Flip the gripper and its camera mount by 180 degrees at episode reset.
INITIAL_ARM_POSE = np.array([.3879075, -.9498327, .2974578, 1.65806, 1.584 - np.pi, .7], dtype=np.float32)
BALL_RADIUS = .022
PLATE_TOP_Z = .005
PLATE_HALF_SIZE = np.array([.08, .08], dtype=np.float32)
PLATE_SAFE_HALF_SIZE = PLATE_HALF_SIZE - BALL_RADIUS - .005
SOURCES = np.array([[.23, -.085, .027]], dtype=np.float32)
GOALS = np.array([[.23, .10, PLATE_TOP_Z + BALL_RADIUS]], dtype=np.float32)
TARGET_ORDER = np.array([0], dtype=np.int64)
BALL_GOALS = GOALS[TARGET_ORDER]
LIFT_HEIGHT = .04
LIFT_STEPS = 5
STABLE_STEPS = 25
PLACE_Z_TOLERANCE = .004
PLACE_SPEED_LIMIT = .03
SHAPING_SCALE = 3.
FIRST_LIFT_REWARD = 3.
SUCCESS_REWARD = 50.
FAILURE_PENALTY = 10.
TIME_PENALTY = .002
ACTION_PENALTY = .001
ACTION_CHANGE_PENALTY = .002
THIRD_PARTY_LICENSES = """Robot model: Google DeepMind MuJoCo Menagerie,
robotstudio_so101, Apache License 2.0. The original model LICENSE is downloaded
with its assets. The task adds two flat pads and one physical free ball.
No scripted picking controller or trajectory code is included in this module.
"""


def _scene_xml(robot_xml):
    root = ET.parse(robot_xml).getroot()
    root.set("model", TASK_ID)
    root.find("compiler").set("meshdir", str(robot_xml.parent / "assets"))
    root.find("option").attrib.update(timestep=".001", iterations="100",
                                      ls_iterations="50", impratio="100", tolerance="1e-6")
    world = root.find("worldbody")
    # A fixed workspace view complements the moving wrist camera.
    ET.SubElement(world, "camera", name="overview_cam", pos=".55 -.45 .55",
                  xyaxes=".8159 .5800 0 -.419 .589 .694", fovy="52")
    ET.SubElement(world, "light", pos=".2 -.2 1.5", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .1", rgba=".9 .92 .95 1")
    ET.SubElement(world, "geom", name="source_pad", type="box",
                  pos=f"{SOURCES[0, 0]} {SOURCES[0, 1]} {PLATE_TOP_Z / 2}",
                  size=f".06 .06 {PLATE_TOP_Z / 2}", rgba=".10 .40 .32 1")
    ET.SubElement(world, "geom", name="plate", type="box",
                  pos=f"{GOALS[0, 0]} {GOALS[0, 1]} {PLATE_TOP_Z / 2}",
                  size=f"{PLATE_HALF_SIZE[0]} {PLATE_HALF_SIZE[1]} {PLATE_TOP_Z / 2}",
                  friction="1.5 .02 .002", condim="6", rgba=".77 .68 .48 1")
    body = ET.SubElement(world, "body", name="ball0",
                         pos=" ".join(str(float(v)) for v in SOURCES[0]))
    ET.SubElement(body, "freejoint", name="ball_free0")
    ET.SubElement(body, "geom", name="ball0", type="sphere", size=str(BALL_RADIUS),
                  mass=".025", friction="1.5 .02 .002", condim="6", rgba=".95 .22 .09 1")
    return ET.tostring(root, encoding="unicode")


class SceneBundle:
    control_dt = CONTROL_DT
    observation_dim = OBSERVATION_DIM
    action_dim = ACTION_DIM
    num_balls = NUM_BALLS

    def __init__(self, asset_dir, max_episode_steps=EPISODE_STEPS,
                 visual_device=None, camera_names=CAMERA_NAMES):
        self.asset_dir = Path(asset_dir).resolve()
        self.episode_steps = int(max_episode_steps)
        if self.episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        robot_xml = self.asset_dir / "robotstudio_so101" / "so101.xml"
        self.xml = _scene_xml(robot_xml)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.visual_device = str(visual_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.camera_names = tuple(camera_names)
        if self.camera_names != CAMERA_NAMES:
            raise ValueError("camera_names must match CAMERA_NAMES and the policy observation specification")
        for name in self.camera_names:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError(f"Policy camera {name!r} is missing from the robot scene")
        self.image_shape = (FRAME_STACK * len(self.camera_names) * 3, IMAGE_SIZE, IMAGE_SIZE)
        self.observation_spec = dict(OBSERVATION_SPEC, camera_names=list(self.camera_names),
                                     image_shape=list(self.image_shape))
        self.lower, self.upper = self.model.actuator_ctrlrange.T.copy()
        self.initial_pose = np.clip(INITIAL_ARM_POSE, self.lower, self.upper).astype(np.float32)
        self.ball_body_ids = np.array([self.model.body(f"ball{k}").id for k in range(NUM_BALLS)])
        self.ball_geom_ids = np.array([self.model.geom(f"ball{k}").id for k in range(NUM_BALLS)])
        self.plate_geom_id = self.model.geom("plate").id
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


def _potential_numpy(balls, aperture, left, right, lifted, released, plate_contacts):
    """Bounded state score; only its discounted change enters the reward."""
    distance = np.linalg.norm(aperture - balls, axis=1)
    goal_distance = np.linalg.norm(balls - BALL_GOALS, axis=1)
    bilateral = left & right
    height = np.clip((balls[:, 2] - SOURCES[:, 2]) / .06, 0, 1)
    inside = (np.abs(balls[:, :2] - BALL_GOALS[:, :2]) <= PLATE_SAFE_HALF_SIZE).all(axis=1)
    settled = (lifted & released & inside & plate_contacts & ~left & ~right)
    return (.5 * (1 - np.tanh(10 * distance)) + .5 * bilateral +
            .75 * height * bilateral + 1.5 * lifted * np.exp(-10 * goal_distance) +
            3 * settled * np.exp(-100 * np.abs(balls[:, 2] - BALL_GOALS[:, 2])) *
            np.clip(distance / .08, 0, 1))


class _CameraStack:
    """Native MJWarp RGB rendering shared by training and CPU-physics viewing.

    Frames are ordered oldest to latest, each with cameras in CAMERA_NAMES order.
    RGB is kept as uint8 so a PPO rollout does not store float32 image copies.
    """
    def __init__(self, bundle, num_envs=1, device=None, model=None, data=None):
        self.bundle, self.num_envs = bundle, int(num_envs)
        self.device = torch.device(device or bundle.visual_device)
        wp.init()
        self.wp_device = str(self.device) if self.device.type == "cuda" else "cpu"
        self.stream = (wp.stream_from_torch(torch.cuda.current_stream(self.device))
                       if self.device.type == "cuda" else None)
        self.closed = False
        with self.scope():
            self.model = mjw.put_model(bundle.model) if model is None else model
            self.data = (mjw.make_data(bundle.model, nworld=self.num_envs, nconmax=192, njmax=600)
                         if data is None else data)
            self.owns_physics = data is None
            if self.owns_physics:
                self.qpos = wp.to_torch(self.data.qpos)
                self.qvel = wp.to_torch(self.data.qvel)
                self.ctrl = wp.to_torch(self.data.ctrl)
            try:
                self.context = mjw.create_render_context(
                    bundle.model, nworld=self.num_envs, cam_res=(IMAGE_SIZE, IMAGE_SIZE),
                    cam_active=list(bundle.camera_names), render_rgb=True,
                    render_depth=False, render_seg=False, use_textures=False,
                    use_shadows=False, samples_per_pixel=1, enabled_geom_groups=[0, 1, 2])
            except Exception as exc:
                self.close()
                raise RuntimeError(
                    "Policy RGB rendering could not start. Use the pinned mujoco-warp "
                    "version and a Colab GPU runtime; image observations cannot be replaced "
                    "with exact simulator object coordinates."
                ) from exc
            self.rgb_buffers = [wp.zeros((self.num_envs, IMAGE_SIZE, IMAGE_SIZE),
                                          dtype=wp.vec3, device=self.wp_device)
                                for _ in bundle.camera_names]
            self.rgb_views = [wp.to_torch(buffer) for buffer in self.rgb_buffers]
            self.frames = torch.zeros(
                (self.num_envs, FRAME_STACK, len(bundle.camera_names), 3, IMAGE_SIZE, IMAGE_SIZE),
                dtype=torch.uint8, device=self.device)

    def scope(self):
        stack = contextlib.ExitStack()
        stack.enter_context(wp.ScopedDevice(self.wp_device))
        if self.stream is not None:
            stack.enter_context(wp.ScopedStream(self.stream))
        return stack

    def refresh(self, update_mask=None, reset_mask=None, cpu_data=None):
        """Render the current physical state; update only requested worlds' caches."""
        if self.closed:
            raise RuntimeError("The policy camera renderer is closed")
        with self.scope():
            if self.owns_physics:
                if cpu_data is None:
                    raise ValueError("CPU physical state is required for the policy cameras")
                self.qpos.copy_(torch.as_tensor(cpu_data.qpos, device=self.device).unsqueeze(0))
                self.qvel.copy_(torch.as_tensor(cpu_data.qvel, device=self.device).unsqueeze(0))
                self.ctrl.copy_(torch.as_tensor(cpu_data.ctrl, device=self.device).unsqueeze(0))
                mjw.forward(self.model, self.data)
            # The caller has already forwarded shared training physics.
            mjw.refit_bvh(self.model, self.data, self.context)
            mjw.render(self.model, self.data, self.context)
            cameras = []
            for index, (buffer, view) in enumerate(zip(self.rgb_buffers, self.rgb_views)):
                # Indices refer to the requested cam_active list, not model camera ids.
                mjw.get_rgb(self.context, index, buffer)
                cameras.append(view.permute(0, 3, 1, 2).mul(255).round().clamp(0, 255).to(torch.uint8))
            current = torch.stack(cameras, dim=1)
            update = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                      if update_mask is None else torch.as_tensor(
                          update_mask, dtype=torch.bool, device=self.device))
            resets = (torch.zeros_like(update) if reset_mask is None else torch.as_tensor(
                reset_mask, dtype=torch.bool, device=self.device))
            shifted = torch.cat((self.frames[:, 1:], current[:, None]), dim=1)
            repeated = current[:, None].expand(-1, FRAME_STACK, -1, -1, -1, -1)
            replacement = torch.where(resets[:, None, None, None, None, None], repeated, shifted)
            self.frames.copy_(torch.where(update[:, None, None, None, None, None],
                                          replacement, self.frames))

    def image(self):
        return self.frames.reshape(self.num_envs, *self.bundle.image_shape)

    def close(self):
        if self.closed:
            return
        with self.scope():
            wp.synchronize_device(self.wp_device)
        self.closed = True
        self.context = None
        self.rgb_views = self.rgb_buffers = None
        self.frames = None
        self.model = self.data = None
        if self.owns_physics:
            self.qpos = self.qvel = self.ctrl = None


class CPUEnv:
    observation_dim = OBSERVATION_DIM
    action_dim = ACTION_DIM
    num_balls = NUM_BALLS

    def __init__(self, bundle, seed=7, jitter=.015, max_episode_steps=None):
        self.bundle, self.model = bundle, bundle.model
        self.episode_steps = int(max_episode_steps or bundle.episode_steps)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)
        self.data = mujoco.MjData(self.model)
        self.observation_spec = dict(bundle.observation_spec)
        self.cameras = _CameraStack(bundle)
        self._camera_image = None
        self.last_info = {}
        try:
            self.reset(seed)
        except Exception:
            self.close()
            raise

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
        self.plate_contacts = np.zeros(NUM_BALLS, dtype=bool)
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.reward_components = {}
        self.max_heights = SOURCES[:, 2].copy()
        self.total_return = 0.
        self.success = self.done = False
        self.failure = ""
        mujoco.mj_forward(self.model, self.data)
        self._refresh_images(reset=True)
        self.last_info = self._info()
        return self.observation()

    def observation(self):
        # Only robot proprioception accompanies RGB. No ball pose, goal pose,
        # contacts or reward bookkeeping is exposed to either actor or critic.
        lower, upper = self.bundle.lower, self.bundle.upper
        proprio = np.concatenate([
            2 * (self.data.qpos[:6] - lower) / (upper - lower) - 1,
            np.clip(self.data.qvel[:6] * .1, -5, 5),
            2 * (self.joint_targets - lower) / (upper - lower) - 1,
            self.previous_action,
            [max(0., 1 - self.step_index / self.episode_steps)],
        ]).astype(np.float32)
        return {"image": self._camera_image.copy(),
                "proprio": np.nan_to_num(proprio, nan=0., posinf=0., neginf=0.)}

    def _refresh_images(self, reset=False):
        self.cameras.refresh(reset_mask=[reset], cpu_data=self.data)
        self._camera_image = self.cameras.image()[0].cpu().numpy().copy()

    def close(self):
        if getattr(self, "cameras", None) is not None:
            self.cameras.close()
            self.cameras = None

    def _jaw_contacts(self):
        """Read force-bearing jaw and plate contacts for the free ball."""
        contacts = np.zeros((NUM_BALLS, 3), dtype=bool)
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
            mujoco.mj_contactForce(self.model, self.data, index, force)
            body = int(self.model.geom_bodyid[other])
            if force[0] > .05:
                contacts[ball, 0] |= body == self.bundle.left_body
                contacts[ball, 1] |= body == self.bundle.right_body
            if force[0] > .02 and other == self.bundle.plate_geom_id:
                contacts[ball, 2] = True
        return contacts[:, 0], contacts[:, 1], contacts[:, 2]

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
            "control": "policy_joint_delta_targets", "task_id": TASK_ID,
            "plate_contacts": self.plate_contacts.tolist(),
            "observation_spec": dict(self.observation_spec),
            "reward_components": dict(self.reward_components),
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
                                   self.right_contacts, self.lifted, self.released,
                                   self.plate_contacts)[active]
        old_lifted = self.lifted.copy()
        action = np.tanh(raw)
        action_change = action - self.previous_action
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
        inside = (np.abs(balls[:, :2] - BALL_GOALS[:, :2]) <= PLATE_SAFE_HALF_SIZE).all(axis=1)
        left, right, plate = self._jaw_contacts()
        self.left_contacts, self.right_contacts, self.plate_contacts = left, right, plate
        self.bilateral_seen |= left & right
        raised = ((balls[:, 2] >= SOURCES[:, 2] + LIFT_HEIGHT) &
                  (distance < .055) & left & right)
        self.lift_steps = np.where(raised, self.lift_steps + 1, 0)
        self.lifted |= self.lift_steps >= LIFT_STEPS
        self.released |= self.lifted & (self.data.qpos[5] > .6) & ~left & ~right & (distance > .045)
        stable = (self.lifted & self.bilateral_seen & self.released & inside & plate &
                  (np.abs(balls[:, 2] - BALL_GOALS[:, 2]) <= PLACE_Z_TOLERANCE) &
                  (speed < PLACE_SPEED_LIMIT) & ~left & ~right & (distance > .06))
        self.stable_steps = np.where(stable, self.stable_steps + 1, 0)
        self.filled = self.stable_steps >= STABLE_STEPS
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        physical_failure = bool(not finite or (balls[:, 2] < -.02).any() or
                                (np.abs(balls[:, :2]) > .65).any())
        self.success = bool(self.filled.all() and not physical_failure)
        self.failure = "nonfinite_or_out_of_workspace" if physical_failure else ""
        terminated = bool(self.success or physical_failure)
        # A true terminal has zero successor potential. Time limits bootstrap
        # from the physical successor and therefore retain its potential.
        after = 0. if terminated else _potential_numpy(
            balls, self.bundle.aperture(self.data), left, right, self.lifted,
            self.released, plate)[active]
        self.reward_components = {
            "progress": float(SHAPING_SCALE * (REWARD_GAMMA * after - before)),
            "first_lift": float(FIRST_LIFT_REWARD * (self.lifted & ~old_lifted).sum()),
            "success": float(SUCCESS_REWARD * self.success),
            "failure": float(-FAILURE_PENALTY * physical_failure),
            "time": -TIME_PENALTY,
            "action": float(-ACTION_PENALTY * np.square(action).mean()),
            "action_change": float(-ACTION_CHANGE_PENALTY * np.square(action_change).mean()),
        }
        reward = float(np.nan_to_num(sum(self.reward_components.values())))
        self.previous_action = action.copy()
        self.rewarded |= self.filled
        self.max_heights = np.maximum(self.max_heights, balls[:, 2])
        self.total_return += reward
        self.step_index += 1
        truncated = bool(self.step_index >= self.episode_steps and not terminated)
        self.done = terminated or truncated
        self.last_info = self._info()
        # Invalid terminal physics cannot be submitted to the renderer's BVH.
        # Its observation retains the last valid image; terminals do not bootstrap.
        if finite and (self.step_index % IMAGE_INTERVAL == 0 or self.done):
            self._refresh_images()
        return self.observation(), reward, terminated, truncated, dict(self.last_info)


@torch.inference_mode()
def evaluate_policy(bundle, policy=None, mode="learned", episodes=1, seed=2026,
                    jitter=.015, max_episode_steps=None):
    """Optional user-run image-policy evaluation; CPU physics and MJWarp RGB."""
    results = []
    rng = np.random.default_rng(seed)
    for episode in range(int(episodes)):
        env = None
        try:
            env = CPUEnv(bundle, seed=seed + episode, jitter=jitter,
                         max_episode_steps=max_episode_steps)
            obs = env.observation()
            for _ in range(env.episode_steps):
                if mode == "zero_action":
                    action = np.zeros(ACTION_DIM)
                elif mode == "random":
                    action = rng.standard_normal(ACTION_DIM)
                elif mode == "learned" and policy is not None:
                    policy_device = next(policy.parameters()).device
                    batched = {key: torch.as_tensor(value, device=policy_device).unsqueeze(0)
                               for key, value in obs.items()}
                    action = policy.mean_action(batched).squeeze(0).cpu().numpy()
                else:
                    raise ValueError("Use learned with a policy, random, or zero_action")
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            results.append(info)
        finally:
            if env is not None:
                env.close()
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
        "backend": "CPU MuJoCo physics / native MuJoCo Warp RGB observations", "mode": mode,
        "task_id": TASK_ID, "observation_spec": dict(bundle.observation_spec),
        "success_contract": "one physically grasped and lifted ball released onto the flat plate, supported and stable for 0.5 seconds",
    }


class WarpEnv:
    """Direct policy actions in batched MuJoCo Warp; only whole episodes reset."""
    observation_dim = OBSERVATION_DIM
    action_dim = ACTION_DIM
    num_balls = NUM_BALLS

    def __init__(self, bundle, num_envs=8, device="cuda:0", seed=7, jitter=.015,
                 capture_graph=False, max_episode_steps=None):
        self.bundle, self.num_envs = bundle, int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.episode_steps = int(max_episode_steps or bundle.episode_steps)
        self.device = torch.device(device)
        self.observation_spec = dict(bundle.observation_spec)
        # Torch stream conversion requires Warp's runtime to exist first.
        wp.init()
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
            self.safe_half_size = torch.as_tensor(PLATE_SAFE_HALF_SIZE, device=self.device)
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
            self.previous_action = torch.zeros_like(self.joint_targets)
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
            self.plate_contacts = torch.zeros_like(self.lifted)
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
            self.cameras = _CameraStack(bundle, self.num_envs, self.device,
                                        model=self.model, data=self.data)
            self.cameras.refresh(reset_mask=torch.ones_like(self.success))

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
        self.previous_action.masked_fill_(mask[:, None], 0)
        self.steps.masked_fill_(mask, 0)
        self.returns.masked_fill_(mask, 0)
        for field in (self.lift_steps, self.stable_steps):
            field.masked_fill_(mask[:, None], 0)
        for field in (self.lifted, self.bilateral_seen, self.released, self.filled,
                      self.rewarded, self.left_contacts, self.right_contacts, self.plate_contacts):
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
        # Task coordinates and contact/success flags remain reward-internal.
        proprio = torch.cat([
            2 * (self.qpos[:, :6] - self.lower) / (self.upper - self.lower) - 1,
            (self.qvel[:, :6] * .1).clamp(-5, 5),
            2 * (self.joint_targets - self.lower) / (self.upper - self.lower) - 1,
            self.previous_action,
            (1 - self.steps[:, None] / self.episode_steps).clamp_min(0),
        ], dim=1)
        return {"image": self.cameras.image(),
                "proprio": torch.nan_to_num(proprio, nan=0., posinf=0., neginf=0.)}

    def _refresh_images(self, update_mask, finite):
        # Keep the last valid frame for nonfinite failure terminals. A render
        # traverses every world's BVH, even when only some image caches update,
        # so invalid worlds need a temporary finite render pose.
        valid_updates = update_mask & finite
        if not bool(valid_updates.any()):
            return
        if bool(finite.all()):
            self.cameras.refresh(update_mask=valid_updates)
            return
        saved_qpos, saved_qvel = self.qpos.clone(), self.qvel.clone()
        safe_qpos = torch.as_tensor(self.bundle.model.qpos0, dtype=self.qpos.dtype,
                                    device=self.device).clone()
        safe_qpos[:ACTION_DIM] = self.initial_pose
        try:
            self.qpos.copy_(torch.where(finite[:, None], self.qpos, safe_qpos[None]))
            self.qvel.copy_(torch.where(finite[:, None], self.qvel, torch.zeros_like(self.qvel)))
            mjw.forward(self.model, self.data)
            self.cameras.refresh(update_mask=valid_updates)
        finally:
            # Preserve terminal proprioception until final_obs is copied.
            # Do not forward invalid state: the caller resets all failed worlds
            # immediately after recording terminal observations and metrics.
            self.qpos.copy_(saved_qpos)
            self.qvel.copy_(saved_qvel)

    def _potential(self):
        balls = self._balls()
        distance = (self._eef()[:, None, :] - balls).norm(dim=2)
        goal_distance = (balls - self.goals).norm(dim=2)
        bilateral = self.left_contacts & self.right_contacts
        height = ((balls[:, :, 2] - self.sources[:, 2]) / .06).clamp(0, 1)
        inside = ((balls[:, :, :2] - self.goals[:, :2]).abs() <= self.safe_half_size).all(dim=2)
        settled = (self.lifted & self.released & inside & self.plate_contacts &
                   ~self.left_contacts & ~self.right_contacts)
        return (.5 * (1 - torch.tanh(10 * distance)) + .5 * bilateral +
                .75 * height * bilateral + 1.5 * self.lifted * torch.exp(-10 * goal_distance) +
                3 * settled * torch.exp(-100 * (balls[:, :, 2] - self.goals[:, 2]).abs()) *
                (distance / .08).clamp(0, 1))

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
                 (wp.to_torch(contact.dist) <= .0005))
        bodies = self.geom_bodyid[other]
        force = self.contact_force_view[:, 0]
        index = world.clamp(0, self.num_envs - 1) * NUM_BALLS + ball.clamp(0, NUM_BALLS - 1)
        matches = [(bodies == self.bundle.left_body) & (force > .05),
                   (bodies == self.bundle.right_body) & (force > .05),
                   (other == self.bundle.plate_geom_id) & (force > .02)]
        result = []
        for match in matches:
            out = torch.zeros(self.num_envs * NUM_BALLS, device=self.device)
            out.scatter_add_(0, index, (valid & match).float())
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
            action_change = action - self.previous_action
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
            inside = ((balls[:, :, :2] - self.goals[:, :2]).abs() <= self.safe_half_size).all(dim=2)
            left, right, plate = self._jaw_contacts()
            self.left_contacts.copy_(left)
            self.right_contacts.copy_(right)
            self.plate_contacts.copy_(plate)
            self.bilateral_seen.logical_or_(left & right)
            raised = ((balls[:, :, 2] >= self.sources[:, 2] + LIFT_HEIGHT) &
                      (distance < .055) & left & right)
            self.lift_steps.copy_(torch.where(raised, self.lift_steps + 1, 0))
            self.lifted.logical_or_(self.lift_steps >= LIFT_STEPS)
            self.released.logical_or_(self.lifted & (self.qpos[:, 5:6] > .6) &
                                      ~left & ~right & (distance > .045))
            stable = (self.lifted & self.bilateral_seen & self.released & inside & plate &
                      ((balls[:, :, 2] - self.goals[:, 2]).abs() <= PLACE_Z_TOLERANCE) &
                      (speed < PLACE_SPEED_LIMIT) & ~left & ~right & (distance > .06))
            self.stable_steps.copy_(torch.where(stable, self.stable_steps + 1, 0))
            self.filled.copy_(self.stable_steps >= STABLE_STEPS)
            finite = torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1)
            physical_failure = (~finite | (balls[:, :, 2] < -.02).any(dim=1) |
                                (balls[:, :, :2].abs() > .65).any(dim=2).any(dim=1))
            self.success.copy_(self.filled.all(dim=1) & ~physical_failure)
            terminated = physical_failure | self.success
            after = self._potential()[self.world, active]
            # A true terminal successor is absorbing; a time limit bootstraps.
            after = torch.where(terminated, torch.zeros_like(after), after)
            reward_components = {
                "progress": SHAPING_SCALE * (REWARD_GAMMA * after - before),
                "first_lift": FIRST_LIFT_REWARD * (self.lifted & ~old_lifted).sum(dim=1),
                "success": SUCCESS_REWARD * self.success,
                "failure": -FAILURE_PENALTY * physical_failure,
                "time": torch.full_like(before, -TIME_PENALTY),
                "action": -ACTION_PENALTY * action.square().mean(dim=1),
                "action_change": -ACTION_CHANGE_PENALTY * action_change.square().mean(dim=1),
            }
            reward = torch.nan_to_num(sum(reward_components.values()), nan=0., posinf=0., neginf=0.)
            self.previous_action.copy_(action)
            self.rewarded.logical_or_(self.filled)
            self.returns.add_(reward)
            self.steps.add_(1)
            truncated = (self.steps >= self.episode_steps) & ~terminated
            done = terminated | truncated
            image_due = (self.steps.remainder(IMAGE_INTERVAL) == 0) | done
            if bool(image_due.any()):
                self._refresh_images(image_due, finite)
            # Copy both fields before automatic resets overwrite camera history.
            final_obs = {key: value.clone() for key, value in self.observation().items()}
            filled_count = self.filled.sum(dim=1)
            grid = torch.zeros_like(self.filled)
            grid[:, self.grid_order] = self.filled
            info = {
                "final_obs": final_obs, "episode_returns": self.returns[done].clone(),
                "episode_success": self.success[done].clone(),
                "episode_lifted": self.lifted.any(dim=1)[done].clone(),
                "episode_released": self.released.any(dim=1)[done].clone(),
                "episode_filled_count": filled_count[done].clone(),
                "episode_xy_error": errors[done].amax(dim=1).clone(),
                "filled_count": filled_count.clone(), "filled_mask": grid.clone(),
                "active_ball_index": active.clone(), "target_cell": self.grid_order[active],
                "reward_components": {key: value.clone() for key, value in reward_components.items()},
                "task_id": TASK_ID,
            }
            if bool(done.any()):
                self._reset(done)
                mjw.forward(self.model, self.data)
                # Every reset world gets only its new episode's initial frame.
                self.cameras.refresh(update_mask=done, reset_mask=done)
            return self.observation(), reward, terminated, truncated, info

    def close(self):
        self.graph = None
        if getattr(self, "cameras", None) is not None:
            self.cameras.close()
            self.cameras = None
