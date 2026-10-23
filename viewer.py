"""Browser visualization of an independent visual-policy rollout.

MuJoCo physics runs on CPU; policy cameras use the same GPU renderer as training.

The RL policy chooses all arm-joint and gripper target increments. This viewer
does not contain a trajectory, an inverse-kinematics planner, or a scripted pick.
"""
from __future__ import annotations

import contextlib
import threading
import time
import traceback

import numpy as np
import torch

from environment import (CPUEnv, GOALS, TARGET_ORDER, PLATE_HALF_SIZE, PLATE_TOP_Z,
                         CAMERA_NAMES, IMAGE_SHAPE)
from learning import ActorCritic, batch_observation


class LiveSO101Viewer:
    """Show one visual-policy rollout with CPU physics and GPU camera rendering.

    Only the worker thread owns its MuJoCo data. GUI callbacks communicate
    through events; copied network weights stay fixed for a whole episode.
    """

    MODES = ("학습 시작 시 정책", "최신 정책", "무작위 행동")

    def __init__(self, runner, bundle, port=8081, seed=2026, jitter=.015):
        import viser
        from mjviser import ViserMujocoScene

        self.runner = runner
        self.bundle = bundle
        self.seed = int(seed)
        self.jitter = float(jitter)
        self.num_balls = int(getattr(bundle, "num_balls", 1))
        self.target_cell = int(TARGET_ORDER[0])
        self.active_ball_index = 0
        self.filled_count = 0
        self.filled_mask = np.zeros(len(GOALS), dtype=bool)
        self.reset_event = threading.Event()
        self.refresh_event = threading.Event()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.selection_lock = threading.Lock()
        self.selection = "최신 정책"
        self.error = None
        self.data = None
        self.last_info = {}
        self.episode_policy_version = None
        self.closed = False
        self.server = viser.ViserServer(host="127.0.0.1", port=port)
        self.scene = ViserMujocoScene(self.server, bundle.model, num_envs=1)
        self.server.scene.set_up_direction("+z")
        self.goal_labels = [self.server.scene.add_label(
            f"/task_overlay/plate_{index + 1}", text="목표 플레이트",
            position=tuple(goal + np.array([0., 0., .045])),
            font_size_mode="screen", font_screen_scale=.7,
            anchor="center-center",
        ) for index, goal in enumerate(GOALS)]
        half_x, half_y = map(float, PLATE_HALF_SIZE)
        vertices = np.array([[-half_x, -half_y, 0.], [half_x, -half_y, 0.],
                             [half_x, half_y, 0.], [-half_x, half_y, 0.]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        self.fill_markers = [self.server.scene.add_mesh_simple(
            f"/task_overlay/filled_{index + 1}", vertices=vertices, faces=faces,
            color=(40, 185, 95), opacity=.35, side="double",
            position=(float(goal[0]), float(goal[1]), float(PLATE_TOP_Z) + .001), visible=False,
        ) for index, goal in enumerate(GOALS)]
        self.target_marker = self.server.scene.add_mesh_simple(
            "/task_overlay/current_goal", vertices=vertices, faces=faces,
            color=(40, 150, 245), opacity=.4, side="double",
            position=(float(GOALS[self.target_cell, 0]),
                      float(GOALS[self.target_cell, 1]), float(PLATE_TOP_Z) + .0015),
        )
        self.goal_labels[self.target_cell].text = "목표 플레이트"
        self.scene.camera_tracking_enabled = False
        self.scene.set_refresh_handler(self.refresh_event.set)
        self.scene.create_visualization_gui(
            camera_distance=1.25, camera_azimuth=130, camera_elevation=30,
        )
        with self.server.gui.add_folder("강화학습과 평가"):
            self.status_text = self.server.gui.add_markdown("정책 평가 화면 준비 중")
            self.select = self.server.gui.add_dropdown(
                "표시할 정책", self.MODES, initial_value="최신 정책",
            )
            self.pause = self.server.gui.add_checkbox(
                "뷰어만 일시정지", initial_value=False,
            )
            self.reset = self.server.gui.add_button("공 1개와 빈 플레이트로 다시 시작")
            self.episode_text = self.server.gui.add_markdown("물리 시뮬레이션 준비 중")
            self.result_text = self.server.gui.add_markdown("완료된 화면 에피소드 없음")
            self.reward_chart = self.server.gui.add_uplot(
                data=(np.array([0.]), np.array([np.nan])),
                series=({}, {"label": "학습 완료 에피소드 평균 보상", "stroke": "#2563eb"}),
                scales={"x": {"time": False}}, height=170,
            )
            self.server.gui.add_markdown(
                "**관절 움직임과 집게 개폐를 PPO 정책이 직접 결정합니다.**\n\n"
                "파란 사각형은 평평한 목표 플레이트, 초록은 공이 안정적으로 안착한 상태입니다. "
                "공을 실제로 잡아 들어 옮기고, 플레이트 위에 놓아야 성공입니다.\n\n"
                "정책이 공을 집지 못하거나 불필요하게 움직일 수 있습니다. "
                "학습 성공 여부는 실제 에피소드 결과로 확인하세요.\n\n"
                "최신 가중치는 다음 전체 에피소드 시작 때 반영합니다. "
                "이 화면의 물리는 CPU에서 별도로 평가하며, 정책 입력 영상은 학습과 같은 GPU 렌더러로 만듭니다. "
                "3D 화면의 파란·초록 표시는 관찰자를 위한 것이며 정책 영상에는 들어가지 않습니다. "
                "뷰어의 제어 스텝은 학습 업데이트 수와 다르며, "
                "일시정지와 초기화는 학습에 영향을 주지 않습니다."
            )
        with self.server.gui.add_folder("정책이 보는 영상"):
            self.server.gui.add_markdown(
                "정책에 실제로 전달한 두 카메라의 최신 RGB 프레임입니다. "
                "각 영상은 64×64 픽셀이며, 직전·현재 두 프레임과 "
                "관절 관련 수치 25개로 움직임을 결정합니다. "
                "영상은 12.5Hz로 갱신합니다. 공의 정확한 좌표는 정책에 전달하지 않습니다."
            )
            self.policy_images = [self.server.gui.add_image(
                np.zeros((IMAGE_SHAPE[1], IMAGE_SHAPE[2], 3), dtype=np.uint8),
                label=("손목 카메라" if name == "wrist_cam" else "고정 카메라"),
                format="png",
            ) for name in CAMERA_NAMES]
        self.pause.on_update(self._set_pause)
        self.select.on_update(self._set_selection)
        self.reset.on_click(lambda _: self.reset_event.set())
        self.thread = threading.Thread(
            target=self._loop_guarded, name="SO101-direct-policy-viewer", daemon=True,
        )
        self.thread.start()

    def _set_pause(self, _):
        (self.pause_event.set if self.pause.value else self.pause_event.clear)()

    def _set_selection(self, _):
        with self.selection_lock:
            self.selection = self.select.value
        self.reset_event.set()

    def _loop_guarded(self):
        try:
            self._loop()
        except Exception:
            self.error = traceback.format_exc()
            self.status_text.content = "**뷰어 평가 오류** — 노트북의 오류 로그를 확인하세요."
            print(self.error)

    def _history(self):
        lock = getattr(self.runner, "lock", None)
        with lock if lock is not None else contextlib.nullcontext():
            return [dict(row) for row in self.runner.history]

    def _update_status(self):
        history = self._history()
        if history:
            rewards = [row.get("train_mean_return") for row in history]
            self.reward_chart.data = (
                np.asarray([row.get("steps", 0) for row in history], dtype=float),
                np.asarray([np.nan if item is None else item for item in rewards], dtype=float),
            )
        completed_updates = int(getattr(self.runner, "completed_updates", 0))
        progress = f"{completed_updates} / {self.runner.config.get('updates', '?')} 업데이트"
        score = getattr(self.runner, "final_evaluation", None)
        score_text = "평가 대기"
        if score:
            measured = score.get("learned", score)
            rate = measured.get("success_rate")
            if rate is not None:
                score_text = f"플레이트 옮기기 성공률 {float(rate):.0%}"
        placed_text = "안착 성공" if self.filled_count else "아직 미완료"
        version_text = (str(self.episode_policy_version)
                        if self.episode_policy_version is not None else "무작위 행동")
        self.status_text.content = (
            f"**{self.runner.status}** · {progress}\n\n"
            f"최근 고정 조건 평가: **{score_text}**\n\n"
            f"화면의 공 1개: **{placed_text}** · 목표: **평평한 플레이트**\n\n"
            f"학습 최신 버전: **{getattr(self.runner, 'snapshot_version', 0)}** · "
            f"화면 에피소드 버전: **{version_text}**"
        )

    @staticmethod
    def _reset(env, seed):
        result = env.reset(seed=seed)
        return result[0] if isinstance(result, tuple) else result

    def _update_policy_images(self, observation):
        """Display the latest camera frames already supplied to the policy."""
        images = np.asarray(observation["image"], dtype=np.uint8)
        latest = images[-3 * len(CAMERA_NAMES):]
        for index, handle in enumerate(self.policy_images):
            handle.image = np.moveaxis(latest[3 * index:3 * (index + 1)], 0, -1).copy()

    def _update_progress(self, info):
        self.num_balls = int(info.get("num_balls", self.num_balls))
        self.active_ball_index = 0
        self.target_cell = 0
        mask = np.asarray(info.get("filled_mask", [False]), dtype=bool).reshape(1)
        self.filled_count = int(info.get("filled_count", int(mask.sum())))
        self.filled_mask = mask.copy()
        self.target_marker.position = (
            float(GOALS[0, 0]), float(GOALS[0, 1]), float(PLATE_TOP_Z) + .0015,
        )
        self.fill_markers[0].visible = bool(mask[0])
        self.goal_labels[0].text = "플레이트 안착 ✓" if mask[0] else "목표 플레이트"
        self.target_marker.visible = not bool(mask[0])

    def _update_episode_text(self, selection, step, env):
        info = self.last_info
        lifted = bool(info.get("all_lifted", [False])[0])
        released = bool(info.get("all_released", [False])[0])
        stable = int(info.get("stable_steps", [0])[0])
        self.episode_text.content = (
            f"**{selection}**\n\n"
            f"들어 올림: **{'완료' if lifted else '대기'}** · "
            f"놓기 이력: **{'있음' if released else '없음'}**\n\n"
            f"안착 유지: **{stable} / 25 제어 스텝** · "
            f"성공: **{'완료' if self.filled_count else '대기'}**\n\n"
            f"뷰어 제어 스텝 {step} / {getattr(self.bundle, 'episode_steps', 1000)} · "
            f"시뮬레이션 {float(env.data.time):.1f}초\n\n"
            "이 제어 스텝은 GPU 학습의 업데이트 수와 별개입니다."
        )

    def _loop(self):
        env = CPUEnv(self.bundle, seed=self.seed, jitter=self.jitter)
        try:
            self._run_rollout(env)
        finally:
            env.close()

    def _run_rollout(self, env):
        self.data = env.data
        policy = ActorCritic(env.observation_dim, env.action_dim).cpu().eval()
        obs = self._reset(env, self.seed)
        rng = np.random.default_rng(self.seed + 1)
        previous_selection = None
        policy_version = -1
        step = 0
        total_reward = 0.
        last_render = 0.
        last_status = 0.
        terminal_hold_until = None
        self.last_info = dict(getattr(env, "last_info", {}))
        self._update_progress(self.last_info)
        self.refresh_event.set()
        while not self.stop_event.is_set():
            begin = time.perf_counter()
            with self.selection_lock:
                selection = self.selection
            if selection != previous_selection:
                self.reset_event.set()
            if self.reset_event.is_set():
                self.reset_event.clear()
                obs = self._reset(env, self.seed)
                self.last_info = dict(getattr(env, "last_info", {}))
                self._update_progress(self.last_info)
                rng = np.random.default_rng(self.seed + 1)
                step, total_reward, terminal_hold_until = 0, 0., None
                self._update_episode_text(selection, step, env)
                self.refresh_event.set()
            paused = self.pause_event.is_set()
            if terminal_hold_until is not None and begin >= terminal_hold_until and not paused:
                obs = self._reset(env, self.seed)
                self.last_info = dict(getattr(env, "last_info", {}))
                self._update_progress(self.last_info)
                rng = np.random.default_rng(self.seed + 1)
                step, total_reward, terminal_hold_until = 0, 0., None
                self._update_episode_text(selection, step, env)
                self.refresh_event.set()
            latest_version = int(getattr(self.runner, "snapshot_version", 0))
            learned_mode = selection in ("학습 시작 시 정책", "최신 정책")
            needs_weights = learned_mode and (
                selection != previous_selection or
                (selection == "최신 정책" and step == 0 and latest_version != policy_version)
            )
            if needs_weights:
                weights, policy_version = self.runner.snapshot(initial=selection == "학습 시작 시 정책")
                if weights is not None:
                    policy.load_state_dict(weights)
            previous_selection = selection
            self.episode_policy_version = policy_version if learned_mode else None
            if not paused and terminal_hold_until is None:
                if selection == "무작위 행동":
                    action = rng.standard_normal(env.action_dim).astype(np.float32)
                else:
                    with torch.inference_mode():
                        action = policy.mean_action(batch_observation(obs, device="cpu")).squeeze(0).numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                total_reward += float(reward)
                self.last_info = dict(info)
                self._update_progress(info)
                self._update_episode_text(selection, step, env)
                if terminated or truncated:
                    success = bool(info.get("success", False))
                    result = "공 1개 플레이트 옮기기 성공" if success else "플레이트 옮기기 미완료"
                    reason = info.get("failure", "")
                    reason_text = f" · 종료 정보 {reason}" if reason else ""
                    version_text = (str(self.episode_policy_version)
                                    if self.episode_policy_version is not None else "무작위 행동")
                    self.result_text.content = (
                        f"직전 결과: **{result}**\n\n"
                        f"누적 보상 {total_reward:.2f} · 평가 시드 {self.seed} · "
                        f"정책 버전 {version_text}{reason_text}"
                    )
                    terminal_hold_until = begin + 2.
                    self.refresh_event.set()
            if self.refresh_event.is_set() or begin - last_render >= 1. / 15.:
                self.scene.update_from_mjdata(env.data)
                self._update_policy_images(obs)
                self.refresh_event.clear()
                last_render = begin
            if begin - last_status >= .5:
                self._update_status()
                last_status = begin
            self.stop_event.wait(max(0., float(self.bundle.control_dt) - (time.perf_counter() - begin)))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        self.thread.join(timeout=3.)
        self.server.stop()
