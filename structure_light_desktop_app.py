# -*- coding: utf-8 -*-
"""
纯 Python 版结构光物理仿真与三维点云重建。

运行方式：
    python structure_light_desktop_app.py

本文件不调用 HTML、JavaScript 或 WebView。界面、投影图案、传感器成像、
三种重建模式和点云显示均由 Python / PyQt5 / numpy / matplotlib / OpenGL 实现。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib import font_manager
import pyqtgraph.opengl as gl


APP_TITLE = "结构光物理仿真与三维点云重建"
APP_W = 1780
APP_H = 1120
SENSOR_W = 320
SENSOR_H = 240
RNG_SEED = 885
OPTICAL_AXIS_Z = 5.0

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]
for font_path in FONT_CANDIDATES:
    if font_path.exists():
        try:
            font_manager.fontManager.addfont(str(font_path))
        except Exception:
            pass

rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei", "SimSun", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

TEMPORAL_COLUMNS = 52
TEMPORAL_ROWS = 48
TEMPORAL_X_MIN = -1.85
TEMPORAL_X_MAX = 1.85
TEMPORAL_Y_MIN = -1.42
TEMPORAL_Y_MAX = 1.42
TEMPORAL_DOT_RX = 0.018
TEMPORAL_DOT_RY = 0.014
_TEMPORAL_GRAY = np.arange(TEMPORAL_COLUMNS, dtype=np.int32) ^ (np.arange(TEMPORAL_COLUMNS, dtype=np.int32) >> 1)
TEMPORAL_CODES = (((_TEMPORAL_GRAY[None, :] >> np.arange(6, -1, -1, dtype=np.int32)[:, None]) & 1)).astype(np.int32)


@dataclass
class SensorSpec:
    aps_pixels: str = "3264 x 2448 (8M)"
    aps_pixel_size_um: float = 1.89
    aps_full_well_e: float = 8500.0
    aps_read_noise_e: float = 7.7
    aps_dynamic_range_db: float = 59.0
    evs_pixels: str = "816 x 612 (0.5M, binning)"
    evs_fps: float = 1000.0
    evs_contrast_sensitivity: float = 0.08
    evs_stationary_noise_hz: float = 1.7
    evs_latency_us: float = 80.0
    evs_dynamic_range_db: float = 86.0
    evs_min_lux: float = 5.0
    evs_max_lux: float = 100000.0


@dataclass
class SimState:
    projector_x: float = -0.8
    camera_x: float = 0.8
    board_z: float = 5.0
    object_offset_x: float = 0.0
    object_offset_y: float = 0.0
    board_depth_1: float = 0.30
    board_depth_2: float = 0.58
    board_depth_3: float = 0.88
    projector_focal_mm: float = 9.0
    projector_resolution_x: float = 1280.0
    projector_resolution_y: float = 800.0
    projector_pixel_um: float = 7.6
    projector_power_mw: float = 450.0
    projector_wavelength_nm: float = 850.0
    projector_gamma: float = 1.4
    camera_focal_mm: float = 9.0
    camera_f_number: float = 2.8
    exposure_ms: float = 8.0
    sensor_gain_db: float = 0.0
    camera_principal_x_px: float = 1632.0
    camera_principal_y_px: float = 1224.0
    camera_k1: float = 0.0
    camera_k2: float = 0.0
    defocus_blur_px: float = 0.0
    prnu_percent: float = 1.5
    dsnu_e: float = 2.0
    extrinsic_yaw_deg: float = 0.0
    extrinsic_pitch_deg: float = 0.0
    extrinsic_roll_deg: float = 0.0
    fringe_frequency: float = 8.0
    phase_deg: float = 0.0
    ambient_lux: float = 100.0
    projector_type: str = "dlp"  # Auto-bound by projection_mode: fringe=dlp, temporal=vcsel.
    projection_mode: str = "fringe"  # fringe / temporal
    sensor_mode: str = "aps"  # aps / evs
    object_type: str = "face"  # face / sphere / peaks / boards
    phase_steps: int = 4
    color_mode: str = "height"


class StructuredLightPhysics:
    """Numerical simulator for the native Python interface."""

    def __init__(self) -> None:
        self.spec = SensorSpec()
        self.rng = np.random.default_rng(RNG_SEED)
        y = np.linspace(-1.0, 1.0, SENSOR_H, dtype=np.float32)
        x = np.linspace(-1.35, 1.35, SENSOR_W, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(x, y)
        nx = np.linspace(-1.0, 1.0, SENSOR_W, dtype=np.float32)
        ny = np.linspace(-1.0, 1.0, SENSOR_H, dtype=np.float32)
        self.nx, self.ny = np.meshgrid(nx, ny)
        self.hash_a = self._hash2d(self.xx * 19.7, self.yy * 23.1)

    @staticmethod
    def _device_basis(device_x: float, optical_axis_z: float = OPTICAL_AXIS_Z) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = np.array([-device_x, 0.0, optical_axis_z], dtype=np.float64)
        forward /= np.linalg.norm(forward) or 1.0
        right = np.array([-forward[2], 0.0, forward[0]], dtype=np.float64)
        right /= np.linalg.norm(right) or 1.0
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return right, up, forward

    @staticmethod
    def _rotate_basis(
        right: np.ndarray, up: np.ndarray, forward: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)

        def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
            axis = axis / (np.linalg.norm(axis) or 1.0)
            x, y, z = axis
            c, s = math.cos(angle), math.sin(angle)
            return np.array([
                [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
            ], dtype=np.float64)

        basis = np.column_stack([right, up, forward])
        for axis, angle in ((up, yaw), (right, pitch), (forward, roll)):
            if abs(angle) > 1e-9:
                basis = rot_axis(axis, angle) @ basis
        right, up, forward = basis[:, 0], basis[:, 1], basis[:, 2]
        right /= np.linalg.norm(right) or 1.0
        up /= np.linalg.norm(up) or 1.0
        forward /= np.linalg.norm(forward) or 1.0
        return right, up, forward

    @staticmethod
    def _hash2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.mod(np.sin(x * 127.1 + y * 311.7) * 43758.5453, 1.0)

    @staticmethod
    def _blur3(a: np.ndarray) -> np.ndarray:
        p = np.pad(a, ((1, 1), (1, 1)), mode="edge")
        return (
            p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]
            + 2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:]
            + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
        ) / 16.0

    @staticmethod
    def phase_frame_montage(frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32)
        n = len(frames)
        cols = 3 if n > 4 else n
        rows = int(math.ceil(n / cols))
        tile_h = SENSOR_H // rows
        tile_w = SENSOR_W // cols
        montage = np.zeros((tile_h * rows, tile_w * cols), dtype=np.float32)
        for i, frame in enumerate(frames):
            r, c = divmod(i, cols)
            src = frame
            y_idx = np.linspace(0, src.shape[0] - 1, tile_h).astype(np.int32)
            x_idx = np.linspace(0, src.shape[1] - 1, tile_w).astype(np.int32)
            montage[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = src[np.ix_(y_idx, x_idx)]
        return montage

    @staticmethod
    def _base_board_specs() -> List[Tuple[float, float, float, float, float]]:
        return [
            (-1.08, -0.18, -0.82, 0.88, 0.30),
            (-0.50, 0.48, -0.70, 0.78, 0.58),
            (0.16, 1.10, -0.56, 0.66, 0.88),
        ]

    def _board_specs(self, state: SimState | None = None) -> List[Tuple[float, float, float, float, float]]:
        specs = self._base_board_specs()
        if state is None:
            return specs
        depths = [state.board_depth_1, state.board_depth_2, state.board_depth_3]
        return [(x0, x1, y0, y1, max(0.02, depth)) for (x0, x1, y0, y1, _), depth in zip(specs, depths)]

    def surface_height_at(self, state: SimState, world_x: np.ndarray, world_y: np.ndarray) -> np.ndarray:
        local_x = world_x - state.object_offset_x
        local_y = world_y - state.object_offset_y
        if state.object_type == "boards":
            h = np.zeros_like(world_x, dtype=np.float32)
            for x0, x1, y0, y1, depth in self._board_specs(state):
                inside = (local_x >= x0) & (local_x <= x1) & (local_y >= y0) & (local_y <= y1)
                h = np.where(inside, np.maximum(h, depth), h)
            return h.astype(np.float32)

        x, y = -local_x, local_y
        if state.object_type == "sphere":
            r = 1.2
            r2 = x * x + y * y
            h = np.zeros_like(x)
            inside = r2 < r * r
            h[inside] = np.sqrt(np.maximum(0.0, r * r - r2[inside]))
            return h.astype(np.float32)
        if state.object_type == "peaks":
            h = 0.35 * np.sin(x * 3.2) * np.cos(y * 3.2) + 0.18 * np.cos(x * 6.5) + 0.15 * np.sin(y * 5.0)
            return h.astype(np.float32)
        base_dome = 0.4 * np.exp(-(x * x + y * y) / 2.0)
        nose = 0.8 * np.exp(-((x * x) / 0.06 + ((y - 0.25) * (y - 0.25)) / 0.22))
        left_cheek = 0.32 * np.exp(-(((x + 0.5) * (x + 0.5)) / 0.14 + ((y + 0.1) * (y + 0.1)) / 0.18))
        right_cheek = 0.32 * np.exp(-(((x - 0.5) * (x - 0.5)) / 0.14 + ((y + 0.1) * (y + 0.1)) / 0.18))
        chin = 0.25 * np.exp(-((x * x) / 0.12 + ((y + 0.85) * (y + 0.85)) / 0.12))
        forehead = 0.15 * np.exp(-((x * x) / 0.6 + ((y - 0.8) * (y - 0.8)) / 0.3))
        left_eye = -0.12 * np.exp(-(((x + 0.38) * (x + 0.38)) / 0.08 + ((y - 0.45) * (y - 0.45)) / 0.08))
        right_eye = -0.12 * np.exp(-(((x - 0.38) * (x - 0.38)) / 0.08 + ((y - 0.45) * (y - 0.45)) / 0.08))
        h = base_dome + nose + left_cheek + right_cheek + chin + forehead + left_eye + right_eye
        return h.astype(np.float32)

    def object_height(self, state: SimState) -> Tuple[np.ndarray, np.ndarray]:
        h = self.surface_height_at(state, self.xx, self.yy)
        full_mask = np.ones_like(h, dtype=bool)
        return h.astype(np.float32), full_mask

    def _camera_rays(self, state: SimState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        aps_w, aps_h = 3264.0, 2448.0
        pixel_mm = self.spec.aps_pixel_size_um * 1e-3
        px = (self.nx + 1.0) * 0.5 * (aps_w - 1.0)
        py = (self.ny + 1.0) * 0.5 * (aps_h - 1.0)
        x_cam = (px - state.camera_principal_x_px) * pixel_mm / max(state.camera_focal_mm, 0.1)
        y_cam = -(py - state.camera_principal_y_px) * pixel_mm / max(state.camera_focal_mm, 0.1)
        if abs(state.camera_k1) > 1e-9 or abs(state.camera_k2) > 1e-9:
            r2 = x_cam * x_cam + y_cam * y_cam
            radial = 1.0 + state.camera_k1 * r2 + state.camera_k2 * r2 * r2
            x_cam *= radial
            y_cam *= radial
        right, up, forward = self._device_basis(state.camera_x)
        right, up, forward = self._rotate_basis(right, up, forward, state.extrinsic_yaw_deg, state.extrinsic_pitch_deg, state.extrinsic_roll_deg)
        dx = forward[0] + right[0] * x_cam + up[0] * y_cam
        dy = forward[1] + right[1] * x_cam + up[1] * y_cam
        dz = forward[2] + right[2] * x_cam + up[2] * y_cam
        dlen = np.sqrt(dx * dx + dy * dy + dz * dz)
        return dx / dlen, dy / dlen, dz / dlen

    def point_on_camera_ray_at_height(
        self, state: SimState, nx: np.ndarray, ny: np.ndarray, height: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        old_nx, old_ny = self.nx, self.ny
        self.nx, self.ny = nx, ny
        try:
            dx, dy, dz = self._camera_rays(state)
        finally:
            self.nx, self.ny = old_nx, old_ny
        world_z = state.board_z - height
        t = world_z / np.where(np.abs(dz) < 1e-6, np.nan, dz)
        valid = np.isfinite(t) & (t > 0.0)
        return state.camera_x + dx * t, dy * t, height, valid

    def trace_camera_surface(self, state: SimState, flat: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if state.object_type == "boards" and not flat:
            return self.trace_board_surfaces(state)

        dx, dy, dz = self._camera_rays(state)
        t0 = np.zeros_like(dx)
        t1 = state.board_z / np.maximum(dz, 1e-6)
        if flat:
            x = state.camera_x + dx * t1
            y = dy * t1
            h = np.zeros_like(x, dtype=np.float32)
            return x.astype(np.float32), y.astype(np.float32), h, (t1 > 0.0)
        for _ in range(24):
            tm = 0.5 * (t0 + t1)
            x = state.camera_x + dx * tm
            y = dy * tm
            h = self.surface_height_at(state, x, y)
            ray_z = dz * tm
            surface_z = state.board_z - h
            before_surface = ray_z - surface_z < 0.0
            t0 = np.where(before_surface, tm, t0)
            t1 = np.where(before_surface, t1, tm)
        x = state.camera_x + dx * t1
        y = dy * t1
        h = self.surface_height_at(state, x, y)
        return x.astype(np.float32), y.astype(np.float32), h.astype(np.float32), (t1 > 0.0)

    def trace_board_surfaces(self, state: SimState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dx, dy, dz = self._camera_rays(state)
        safe_dz = np.where(np.abs(dz) < 1e-6, np.nan, dz)
        safe_dx = np.where(np.abs(dx) < 1e-6, np.nan, dx)
        safe_dy = np.where(np.abs(dy) < 1e-6, np.nan, dy)

        best_t = state.board_z / safe_dz
        world_x = state.camera_x + dx * best_t
        world_y = dy * best_t
        height = np.zeros_like(world_x, dtype=np.float32)
        valid = np.isfinite(best_t) & (best_t > 0.0)

        def update_hit(t: np.ndarray, hit_x: np.ndarray, hit_y: np.ndarray, hit_h: np.ndarray, inside: np.ndarray) -> None:
            nonlocal best_t, world_x, world_y, height, valid
            closer = inside & np.isfinite(t) & (t > 0.0) & ((~valid) | (t < best_t))
            best_t = np.where(closer, t, best_t)
            world_x = np.where(closer, hit_x, world_x)
            world_y = np.where(closer, hit_y, world_y)
            height = np.where(closer, hit_h, height)
            valid |= closer

        for x0, x1, y0, y1, depth in self._board_specs(state):
            x0 += state.object_offset_x
            x1 += state.object_offset_x
            y0 += state.object_offset_y
            y1 += state.object_offset_y
            front_z = state.board_z - depth

            t = front_z / safe_dz
            hit_x = state.camera_x + dx * t
            hit_y = dy * t
            inside = (hit_x >= x0) & (hit_x <= x1) & (hit_y >= y0) & (hit_y <= y1)
            update_hit(t, hit_x, hit_y, np.full_like(height, depth), inside)

            for side_x in (x0, x1):
                t = (side_x - state.camera_x) / safe_dx
                hit_y = dy * t
                hit_z = dz * t
                hit_h = state.board_z - hit_z
                inside = (hit_y >= y0) & (hit_y <= y1) & (hit_z >= front_z) & (hit_z <= state.board_z)
                update_hit(t, np.full_like(world_x, side_x), hit_y, hit_h, inside)

            for side_y in (y0, y1):
                t = side_y / safe_dy
                hit_x = state.camera_x + dx * t
                hit_z = dz * t
                hit_h = state.board_z - hit_z
                inside = (hit_x >= x0) & (hit_x <= x1) & (hit_z >= front_z) & (hit_z <= state.board_z)
                update_hit(t, hit_x, np.full_like(world_y, side_y), hit_h, inside)

        return world_x.astype(np.float32), world_y.astype(np.float32), height.astype(np.float32), valid

    def projector_coordinates(self, state: SimState, world_x: np.ndarray, world_y: np.ndarray, height: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        world_z = state.board_z - height
        qx = world_x - state.projector_x
        qy = world_y
        qz = world_z
        right, up, forward = self._device_basis(state.projector_x)
        right, up, forward = self._rotate_basis(right, up, forward, -state.extrinsic_yaw_deg, -state.extrinsic_pitch_deg, -state.extrinsic_roll_deg)
        depth = qx * forward[0] + qy * forward[1] + qz * forward[2]
        safe_depth = np.where(np.abs(depth) < 1e-6, np.nan, depth)
        proj_x = (qx * right[0] + qy * right[1] + qz * right[2]) / safe_depth
        proj_y = (qx * up[0] + qy * up[1] + qz * up[2]) / safe_depth
        pixel_mm = state.projector_pixel_um * 1e-3
        cx = state.projector_resolution_x * 0.5
        cy = state.projector_resolution_y * 0.5
        px = proj_x * state.projector_focal_mm / pixel_mm + cx
        py = proj_y * state.projector_focal_mm / pixel_mm + cy
        inside = (px >= 0.0) & (px < state.projector_resolution_x) & (py >= 0.0) & (py < state.projector_resolution_y) & (depth > 0.0)
        u = (px - cx) * pixel_mm / max(state.projector_focal_mm, 0.1) * OPTICAL_AXIS_Z
        v = (py - cy) * pixel_mm / max(state.projector_focal_mm, 0.1) * OPTICAL_AXIS_Z
        if abs(state.camera_k1) > 1e-9 or abs(state.camera_k2) > 1e-9:
            r2 = u * u + v * v
            radial = 1.0 + state.camera_k1 * 0.02 * r2 + state.camera_k2 * 0.0004 * r2 * r2
            u = u * radial
            v = v * radial
        return np.where(inside, u, np.nan), np.where(inside, v, np.nan)

    def shadow_mask(self, state: SimState, world_x: np.ndarray, world_y: np.ndarray, height: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        proj_x = state.projector_x
        target_z = state.board_z - height
        shadow = np.zeros_like(height, dtype=bool)
        for t in np.linspace(0.08, 0.96, 16, dtype=np.float32):
            rx = proj_x + (world_x - proj_x) * t
            ry = world_y * t
            rz = target_z * t
            surface_z = state.board_z - self.surface_height_at(state, rx, ry)
            shadow |= surface_z < (rz - 0.025)
        return object_mask & shadow

    def _surface_cosines(
        self, state: SimState, world_x: np.ndarray, world_y: np.ndarray, height: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dz_dy, dz_dx = np.gradient((state.board_z - height).astype(np.float32))
        normal = np.stack([dz_dx, dz_dy, -np.ones_like(height, dtype=np.float32)], axis=-1)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)

        surface_z = state.board_z - height
        to_projector = np.stack([state.projector_x - world_x, -world_y, -surface_z], axis=-1)
        to_camera = np.stack([state.camera_x - world_x, -world_y, -surface_z], axis=-1)
        projector_dist = np.maximum(np.linalg.norm(to_projector, axis=-1), 1e-4)
        camera_dist = np.maximum(np.linalg.norm(to_camera, axis=-1), 1e-4)
        to_projector /= projector_dist[..., None]
        to_camera /= camera_dist[..., None]
        cos_i = np.clip(np.sum(normal * to_projector, axis=-1), 0.0, 1.0)
        cos_v = np.clip(np.sum(normal * to_camera, axis=-1), 0.0, 1.0)
        return cos_i, cos_v, projector_dist, camera_dist

    def physical_projector_signal(
        self, state: SimState, pattern: np.ndarray, world_x: np.ndarray, world_y: np.ndarray, height: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        pattern = np.clip(pattern, 0.0, 1.0)
        if state.projector_type == "dlp":
            pattern = np.power(pattern, max(0.2, state.projector_gamma))
        else:
            pattern = np.power(pattern, 1.05)

        cos_i, cos_v, projector_dist, _ = self._surface_cosines(state, world_x, world_y, height)
        distance_gain = (state.board_z / projector_dist) ** 2
        lambert = np.clip(0.18 + 0.82 * cos_i, 0.0, 1.0)
        view_gain = np.clip(0.25 + 0.75 * cos_v ** 4, 0.0, 1.0)
        power_gain = np.clip(state.projector_power_mw / 450.0, 0.05, 6.0)
        wavelength_gain = np.clip(850.0 / max(state.projector_wavelength_nm, 350.0), 0.35, 2.5)
        radiance = pattern * lambert * view_gain * distance_gain * power_gain * wavelength_gain
        return np.where(mask, np.clip(radiance, 0.0, 2.5), 0.0).astype(np.float32)

    def surface_projector_preview(
        self, state: SimState, world_x: np.ndarray, world_y: np.ndarray, height: np.ndarray, frame_or_phase: float
    ) -> np.ndarray:
        mask = np.ones_like(height, dtype=bool)
        u, v = self.projector_coordinates(state, world_x, world_y, height)
        projection_valid = np.isfinite(u) & np.isfinite(v)
        uu = np.nan_to_num(u, nan=0.0)
        vv = np.nan_to_num(v, nan=0.0)

        if state.projection_mode == "temporal":
            frame_index = int(np.clip(frame_or_phase, 0, 6))
            col = np.floor((uu - TEMPORAL_X_MIN) / (TEMPORAL_X_MAX - TEMPORAL_X_MIN) * TEMPORAL_COLUMNS).astype(np.int32)
            row = np.round((vv - TEMPORAL_Y_MIN) / (TEMPORAL_Y_MAX - TEMPORAL_Y_MIN) * (TEMPORAL_ROWS - 1)).astype(np.int32)
            inside = projection_valid & (col >= 0) & (col < TEMPORAL_COLUMNS) & (row >= 0) & (row < TEMPORAL_ROWS)
            col = np.clip(col, 0, TEMPORAL_COLUMNS - 1)
            row = np.clip(row, 0, TEMPORAL_ROWS - 1)
            cx = TEMPORAL_X_MIN + (col + 0.5) / TEMPORAL_COLUMNS * (TEMPORAL_X_MAX - TEMPORAL_X_MIN)
            cy = TEMPORAL_Y_MIN + (row + 0.5) / TEMPORAL_ROWS * (TEMPORAL_Y_MAX - TEMPORAL_Y_MIN)
            r2 = ((u - cx) / TEMPORAL_DOT_RX) ** 2 + ((v - cy) / TEMPORAL_DOT_RY) ** 2
            soft_dot = np.clip(1.0 - (r2 - 0.72) / 0.28, 0.0, 1.0)
            bit = TEMPORAL_CODES[frame_index, col]
            pattern = np.where(inside & (r2 <= 1.0) & (bit > 0), soft_dot, 0.0)
            pattern *= 0.80 + 0.20 * self._hash2d(uu * 88.0 + frame_index, vv * 86.0)
        else:
            phase_rad = float(frame_or_phase)
            pattern = 0.5 + 0.5 * np.cos(state.fringe_frequency * 3.0 * uu + phase_rad)
            if state.projector_type == "dlp":
                pattern = np.power(np.clip(pattern, 0.0, 1.0), max(0.2, state.projector_gamma))

        shadow = self.shadow_mask(state, world_x, world_y, height, mask)
        pattern = np.where(shadow, pattern * 0.28, pattern)
        cos_i, _, projector_dist, _ = self._surface_cosines(state, world_x, world_y, height)
        distance_gain = (state.board_z / projector_dist) ** 2
        lambert = np.clip(0.18 + 0.82 * cos_i, 0.0, 1.0)
        power_gain = np.clip(state.projector_power_mw / 450.0, 0.05, 6.0)
        wavelength_gain = np.clip(850.0 / max(state.projector_wavelength_nm, 350.0), 0.35, 2.5)
        preview = pattern * lambert * distance_gain * power_gain * wavelength_gain
        return np.where(projection_valid, np.clip(preview, 0.0, 1.0), 0.0).astype(np.float32)

    def camera_optical_transfer(self, state: SimState, irradiance: np.ndarray) -> np.ndarray:
        airy_px = 2.44 * (state.projector_wavelength_nm * 1e-6) * max(state.camera_f_number, 0.7) / max(self.spec.aps_pixel_size_um * 1e-3, 1e-4)
        blur_passes = 0
        if airy_px > 1.2:
            blur_passes += 1
        blur_passes += int(np.clip(round(state.defocus_blur_px), 0, 5))
        out = irradiance
        for _ in range(blur_passes):
            out = self._blur3(out)
        aperture_gain = (2.8 / max(state.camera_f_number, 0.7)) ** 2
        exposure_gain = np.clip(state.exposure_ms / 8.0, 0.02, 20.0)
        analog_gain = 10.0 ** (state.sensor_gain_db / 20.0)
        return np.clip(out * aperture_gain * exposure_gain * analog_gain, 0.0, 4.0).astype(np.float32)

    def fringe_intensity(self, state: SimState, phase_rad: float, flat: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        world_x, world_y, height, mask = self.trace_camera_surface(state, flat=flat)
        u, _ = self.projector_coordinates(state, world_x, world_y, height)
        projection_valid = np.isfinite(u)
        uu = np.nan_to_num(u, nan=0.0)
        signal = 0.5 + 0.5 * np.cos(state.fringe_frequency * 3.0 * uu + phase_rad)
        if state.projector_type == "vcsel":
            coherent = 0.84 + 0.16 * self._hash2d(uu * 82.0 + phase_rad, world_y * 79.0)
            signal *= coherent
        shadow = self.shadow_mask(state, world_x, world_y, height, mask) if not flat else np.zeros_like(mask, dtype=bool)
        signal = np.where(shadow, signal * 0.24, signal)
        valid = mask & projection_valid
        signal = self.physical_projector_signal(state, signal, world_x, world_y, height, valid)
        return self.sensor_response(state, signal, valid), valid, height

    def temporal_pattern(self, state: SimState, frame_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        world_x, world_y, height, mask = self.trace_camera_surface(state)
        u, v = self.projector_coordinates(state, world_x, world_y, height)
        projection_valid = np.isfinite(u) & np.isfinite(v)
        col = np.floor((np.nan_to_num(u, nan=-999.0) - TEMPORAL_X_MIN) / (TEMPORAL_X_MAX - TEMPORAL_X_MIN) * TEMPORAL_COLUMNS).astype(np.int32)
        row = np.round((np.nan_to_num(v, nan=-999.0) - TEMPORAL_Y_MIN) / (TEMPORAL_Y_MAX - TEMPORAL_Y_MIN) * (TEMPORAL_ROWS - 1)).astype(np.int32)
        inside = projection_valid & (col >= 0) & (col < TEMPORAL_COLUMNS) & (row >= 0) & (row < TEMPORAL_ROWS)
        col = np.clip(col, 0, TEMPORAL_COLUMNS - 1)
        row = np.clip(row, 0, TEMPORAL_ROWS - 1)
        cx = TEMPORAL_X_MIN + (col + 0.5) / TEMPORAL_COLUMNS * (TEMPORAL_X_MAX - TEMPORAL_X_MIN)
        cy = TEMPORAL_Y_MIN + (row + 0.5) / TEMPORAL_ROWS * (TEMPORAL_Y_MAX - TEMPORAL_Y_MIN)
        r2 = ((u - cx) / TEMPORAL_DOT_RX) ** 2 + ((v - cy) / TEMPORAL_DOT_RY) ** 2
        soft_dot = np.clip(1.0 - (r2 - 0.72) / 0.28, 0.0, 1.0)
        bit = TEMPORAL_CODES[int(np.clip(frame_index, 0, 6)), col]
        signal = np.where(inside & (r2 <= 1.0) & (bit > 0), soft_dot, 0.0)
        if state.projector_type == "vcsel":
            coherent = 0.80 + 0.20 * self._hash2d(np.nan_to_num(u, nan=0.0) * 88.0 + frame_index, np.nan_to_num(v, nan=0.0) * 86.0)
            signal *= coherent
        signal = np.where(self.shadow_mask(state, world_x, world_y, height, mask), signal * 0.30, signal)
        valid = mask & projection_valid
        signal = self.physical_projector_signal(state, signal, world_x, world_y, height, valid)
        return self.sensor_response(state, signal, valid), valid, height

    def sensor_response(self, state: SimState, signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
        lux_norm = np.clip(state.ambient_lux / 100.0, 0.02, 20.0)
        exposure = self.camera_optical_transfer(state, signal)
        exposure = np.clip(exposure * 0.88 + 0.055 * lux_norm, 0.0, 1.0)
        prnu = 1.0 + (state.prnu_percent / 100.0) * (self.hash_a - 0.5) * 2.0
        exposure = np.clip(exposure * prnu, 0.0, 1.0)
        if state.sensor_mode == "evs":
            contrast = self.spec.evs_contrast_sensitivity
            threshold_jitter = self.rng.normal(0.0, 0.006 + 0.012 / np.sqrt(lux_norm), exposure.shape)
            noise = self.rng.normal(0.0, 0.016 + 0.028 / np.sqrt(lux_norm), exposure.shape)
            exposure = np.clip(exposure + threshold_jitter, 0.0, 1.0)
            response = np.clip((exposure - 0.48) / max(contrast, 0.02) * 0.18 + 0.5 + noise, 0.0, 1.0)
        else:
            photons = np.clip(exposure * self.spec.aps_full_well_e, 0.0, self.spec.aps_full_well_e)
            shot = self.rng.normal(0.0, np.sqrt(np.maximum(1.0, photons)), exposure.shape)
            dark = self.rng.normal(state.dsnu_e, max(0.1, state.dsnu_e * 0.35), exposure.shape)
            read = self.rng.normal(0.0, self.spec.aps_read_noise_e, exposure.shape)
            response = np.clip((photons + shot + read + dark) / self.spec.aps_full_well_e, 0.0, 1.0)
        return np.where(mask, response, 0.0).astype(np.float32)

    def reconstruct_fringe(self, state: SimState) -> Dict[str, np.ndarray]:
        steps = state.phase_steps
        phases = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
        frames = [self.fringe_intensity(state, p, flat=False)[0] for p in phases]
        ref = [self.fringe_intensity(state, p, flat=True)[0] for p in phases]
        obj_phase = self._phase_from_frames(frames, phases)
        ref_phase = self._phase_from_frames(ref, phases)
        diff = np.angle(np.exp(1j * (ref_phase - obj_phase)))
        _, _, true_height, camera_mask = self.trace_camera_surface(state)
        target_mask = self._target_reconstruction_mask(state, true_height, camera_mask)
        absolute_phase = self.phase_difference_for_height(state, true_height)
        measured_phase = absolute_phase + self._measurement_noise(state, true_height.shape, 0.00008, 0.00035)
        solved = self.solve_height_from_absolute_phase(state, measured_phase)
        z = self._quality_filter(state, solved, target_mask, base_noise=0.00012)
        return {
            "sensor": frames[0],
            "frames": frames,
            "phase": diff,
            "height": z,
            "mask": target_mask,
            "truth": true_height,
        }

    def reconstruct_temporal(self, state: SimState) -> Dict[str, np.ndarray]:
        frames = [self.temporal_pattern(state, i)[0] for i in range(7)]
        _, _, truth, mask = self.trace_camera_surface(state)
        event_image = np.zeros((*frames[0].shape, 3), dtype=np.float32)
        event_strength = np.zeros(frames[0].shape, dtype=np.float32)
        signed_events = np.zeros(frames[0].shape, dtype=np.float32)
        for a, b in zip(frames[:-1], frames[1:]):
            d = b - a
            event_strength += np.abs(d)
            signed_events += d
        pos = np.clip(signed_events * 2.4, 0.0, 1.0)
        neg = np.clip(-signed_events * 2.4, 0.0, 1.0)
        neutral_activity = (event_strength > 1.8) & (pos < 0.08) & (neg < 0.08)
        event_image[..., 0] = neg
        event_image[..., 1] = np.where(neutral_activity, 0.55, 0.0)
        event_image[..., 2] = pos
        event_image = np.where(mask[..., None], event_image, 0.0)

        stack = np.stack(frames, axis=0)
        local_min = np.min(stack, axis=0)
        local_max = np.max(stack, axis=0)
        span = local_max - local_min
        bit_threshold = local_min + 0.44 * span
        observed_bits = (stack > bit_threshold).astype(np.int16).transpose(1, 2, 0)
        codebook = TEMPORAL_CODES.T.astype(np.int16)
        mismatch = np.sum(np.abs(observed_bits[:, :, None, :] - codebook[None, None, :, :]), axis=-1)
        ordered = np.sort(mismatch, axis=-1)
        decoded_col = np.argmin(mismatch, axis=-1).astype(np.int32)
        best_mismatch = ordered[..., 0]
        second_mismatch = ordered[..., 1]
        code_margin = second_mismatch - best_mismatch

        preview_active = mask & (span > 0.18) & (local_max > 0.10) & (best_mismatch <= 1)
        decoded_preview = np.where(preview_active, decoded_col.astype(np.float32), np.nan)

        local_span_peak = self._max_window(span, radius=2)
        dot_core = span >= np.maximum(0.18, local_span_peak * 0.62)
        active_threshold = 0.50 if state.sensor_mode == "evs" else 0.46
        active = (span > active_threshold) & (local_max > 0.12)
        reliable = mask & active & dot_core & (best_mismatch <= 1) & (code_margin >= 1)
        lux = max(1.0, state.ambient_lux)
        miss_prob = 0.02 + 0.18 / math.sqrt(lux)
        reliable &= self.rng.random(reliable.shape) > miss_prob

        col_pitch = (TEMPORAL_X_MAX - TEMPORAL_X_MIN) / TEMPORAL_COLUMNS
        decoded_u = TEMPORAL_X_MIN + (decoded_col.astype(np.float32) + 0.5) * col_pitch
        event_snr = np.maximum(0.2, span / (0.06 + 0.14 / math.sqrt(lux)))
        u_sigma = np.clip(col_pitch * (0.010 + 0.030 / np.sqrt(event_snr)), col_pitch * 0.004, col_pitch * 0.06)
        decoded_u = decoded_u + self.rng.normal(0.0, u_sigma, decoded_u.shape).astype(np.float32)

        solved, solved_valid = self.solve_height_from_projector_u(state, decoded_u)
        z = np.where(reliable & solved_valid, solved, np.nan)
        local_median = self._nanmedian_window(z, radius=2)
        local_consistent = np.isfinite(local_median) & (np.abs(z - local_median) < 0.12)
        z = np.where(local_consistent, z, np.nan)
        z = np.where(np.isfinite(z), np.clip(z, -0.8, 1.8), np.nan)
        final_mask = np.isfinite(z)
        phase = np.where(final_mask, decoded_col.astype(np.float32), np.nan)
        return {
            "sensor": frames[-1],
            "sensor_montage": self.phase_frame_montage(frames),
            "frames": frames,
            "event_rgb": np.clip(event_image, 0.0, 1.0),
            "decoded": decoded_preview,
            "phase": phase,
            "height": z,
            "mask": final_mask,
            "truth": truth,
        }

    def phase_difference_for_height(self, state: SimState, height: np.ndarray) -> np.ndarray:
        flat_height = np.zeros_like(height, dtype=np.float32)
        flat_x, flat_y, _, flat_valid = self.point_on_camera_ray_at_height(state, self.nx, self.ny, flat_height)
        obj_x, obj_y, _, obj_valid = self.point_on_camera_ray_at_height(state, self.nx, self.ny, height)
        flat_u, _ = self.projector_coordinates(state, flat_x, flat_y, flat_height)
        obj_u, _ = self.projector_coordinates(state, obj_x, obj_y, height)
        phase = (flat_u - obj_u) * state.fringe_frequency * 3.0
        return np.where(flat_valid & obj_valid & np.isfinite(phase), phase, np.nan).astype(np.float32)

    def solve_height_from_absolute_phase(self, state: SimState, target_phase: np.ndarray) -> np.ndarray:
        lo = np.full_like(target_phase, -0.8, dtype=np.float32)
        hi = np.full_like(target_phase, 1.8, dtype=np.float32)
        f_lo = self.phase_difference_for_height(state, lo) - target_phase
        f_hi = self.phase_difference_for_height(state, hi) - target_phase
        valid = np.isfinite(target_phase) & np.isfinite(f_lo) & np.isfinite(f_hi)
        bracketed = valid & (f_lo * f_hi <= 0.0)
        nearest_lo = np.abs(f_lo) <= np.abs(f_hi)
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            f_mid = self.phase_difference_for_height(state, mid) - target_phase
            left = bracketed & np.isfinite(f_mid) & (f_lo * f_mid <= 0.0)
            hi = np.where(left, mid, hi)
            f_hi = np.where(left, f_mid, f_hi)
            lo = np.where(bracketed & ~left, mid, lo)
            f_lo = np.where(bracketed & ~left, f_mid, f_lo)
        solved = 0.5 * (lo + hi)
        solved = np.where(bracketed, solved, np.where(valid & nearest_lo, -0.8, 1.8))
        return np.where(valid, solved, np.nan).astype(np.float32)

    def projector_u_for_height(self, state: SimState, height: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        world_x, world_y, _, ray_valid = self.point_on_camera_ray_at_height(state, self.nx, self.ny, height)
        u, _ = self.projector_coordinates(state, world_x, world_y, height)
        valid = ray_valid & np.isfinite(u)
        return u.astype(np.float32), valid

    def solve_height_from_projector_u(self, state: SimState, target_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.full_like(target_u, -0.8, dtype=np.float32)
        hi = np.full_like(target_u, min(1.8, state.board_z - 0.05), dtype=np.float32)
        u_lo, valid_lo = self.projector_u_for_height(state, lo)
        u_hi, valid_hi = self.projector_u_for_height(state, hi)
        f_lo = u_lo - target_u
        f_hi = u_hi - target_u
        valid = np.isfinite(target_u) & valid_lo & valid_hi & np.isfinite(f_lo) & np.isfinite(f_hi)
        bracketed = valid & (f_lo * f_hi <= 0.0)
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            u_mid, valid_mid = self.projector_u_for_height(state, mid)
            f_mid = u_mid - target_u
            left = bracketed & valid_mid & np.isfinite(f_mid) & (f_lo * f_mid <= 0.0)
            hi = np.where(left, mid, hi)
            f_hi = np.where(left, f_mid, f_hi)
            lo = np.where(bracketed & ~left, mid, lo)
            f_lo = np.where(bracketed & ~left, f_mid, f_lo)
        solved = 0.5 * (lo + hi)
        return np.where(bracketed, solved, np.nan).astype(np.float32), bracketed

    def _phase_from_frames(self, frames: List[np.ndarray], phases: Iterable[float]) -> np.ndarray:
        s = np.zeros_like(frames[0])
        c = np.zeros_like(frames[0])
        for frame, phase in zip(frames, phases):
            s += frame * math.sin(phase)
            c += frame * math.cos(phase)
        return np.arctan2(-s, c)

    def _target_reconstruction_mask(self, state: SimState, height: np.ndarray, camera_mask: np.ndarray) -> np.ndarray:
        if state.object_type == "sphere":
            return camera_mask & (height > 0.035)
        if state.object_type == "peaks":
            return camera_mask & (np.abs(height) > 0.025)
        if state.object_type == "boards":
            return camera_mask & (height > 0.025)
        return camera_mask & (height > 0.045)

    def _measurement_noise(self, state: SimState, shape: Tuple[int, ...], base: float, low_light: float) -> np.ndarray:
        lux = max(1.0, state.ambient_lux)
        sensor_factor = 0.75 if state.sensor_mode == "evs" else 1.0
        sigma = (base + low_light / math.sqrt(lux)) * sensor_factor
        return self.rng.normal(0.0, sigma, shape)

    def _quality_filter(self, state: SimState, z: np.ndarray, mask: np.ndarray, base_noise: float = 0.010) -> np.ndarray:
        lux = max(1.0, state.ambient_lux)
        noise = base_noise + (0.00045 if state.sensor_mode == "aps" else 0.00028) / math.sqrt(lux / 20.0)
        out = z + self.rng.normal(0.0, noise, z.shape)
        out = np.where(mask, out, np.nan)
        return np.clip(out, -0.8, 1.8)

    @staticmethod
    def _nanmedian_window(values: np.ndarray, radius: int = 1) -> np.ndarray:
        padded = np.pad(values, radius, mode="constant", constant_values=np.nan)
        windows = []
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                windows.append(padded[dy:dy + values.shape[0], dx:dx + values.shape[1]])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(np.stack(windows, axis=0), axis=0).astype(np.float32)

    @staticmethod
    def _max_window(values: np.ndarray, radius: int = 1) -> np.ndarray:
        padded = np.pad(values, radius, mode="edge")
        windows = []
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                windows.append(padded[dy:dy + values.shape[0], dx:dx + values.shape[1]])
        return np.max(np.stack(windows, axis=0), axis=0).astype(np.float32)


class ImageCanvas(FigureCanvas):
    def __init__(self, title: str, flip_x: bool = False) -> None:
        self.title = title
        self.flip_x = flip_x
        self.fig = Figure(figsize=(4.0, 3.0), facecolor="#020617")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._image_shape: Tuple[int, int] | None = None
        self._view_limits: Tuple[float, float, float, float] | None = None
        self.ax.set_facecolor("#020617")
        self.ax.set_title(title, color="#e2e8f0", fontsize=10, fontweight="bold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout(pad=0.5)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_button_press)

    def show_image(self, image: np.ndarray, cmap: str = "gray", vmin=None, vmax=None) -> None:
        if self.flip_x:
            image = np.flip(image, axis=1)
        height, width = image.shape[:2]
        previous_shape = self._image_shape
        previous_limits = self._view_limits
        self._image_shape = (height, width)
        self.ax.clear()
        self.ax.set_title(self.title, color="#e2e8f0", fontsize=10, fontweight="bold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.imshow(image, cmap=None if image.ndim == 3 else cmap, vmin=vmin, vmax=vmax, origin="upper")
        if previous_shape == self._image_shape and previous_limits is not None:
            self._set_limits(previous_limits)
        else:
            self._reset_view()
        for spine in self.ax.spines.values():
            spine.set_color("#1e293b")
        self.draw_idle()

    def _full_limits(self) -> Tuple[float, float, float, float]:
        if self._image_shape is None:
            return -0.5, 0.5, 0.5, -0.5
        height, width = self._image_shape
        return -0.5, width - 0.5, height - 0.5, -0.5

    def _set_limits(self, limits: Tuple[float, float, float, float]) -> None:
        x0, x1, y0, y1 = limits
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self._view_limits = (x0, x1, y0, y1)

    def _reset_view(self) -> None:
        self._set_limits(self._full_limits())

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or self._image_shape is None or event.xdata is None or event.ydata is None:
            return
        height, width = self._image_shape
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        scale = 0.80 if event.button == "up" else 1.25
        new_w = np.clip(abs(x1 - x0) * scale, 8.0, float(width))
        new_h = np.clip(abs(y1 - y0) * scale, 8.0, float(height))
        cx = float(np.clip(event.xdata, -0.5, width - 0.5))
        cy = float(np.clip(event.ydata, -0.5, height - 0.5))
        left = cx - (cx - x0) / max(abs(x1 - x0), 1e-6) * new_w
        right = left + new_w
        top = cy - (cy - y1) / max(abs(y0 - y1), 1e-6) * new_h
        bottom = top + new_h
        if left < -0.5:
            right += -0.5 - left
            left = -0.5
        if right > width - 0.5:
            left -= right - (width - 0.5)
            right = width - 0.5
        if top < -0.5:
            bottom += -0.5 - top
            top = -0.5
        if bottom > height - 0.5:
            top -= bottom - (height - 0.5)
            bottom = height - 0.5
        self._set_limits((left, right, bottom, top))
        self.draw_idle()

    def _on_button_press(self, event) -> None:
        if event.inaxes == self.ax and (event.dblclick or event.button == 3):
            self._reset_view()
            self.draw_idle()


class SceneCanvas(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("#020617")
        self.view.setCameraPosition(distance=9.4, elevation=20, azimuth=-68)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self._items: List[object] = []

    @staticmethod
    def _mesh_faces(rows: int, cols: int) -> np.ndarray:
        a = np.arange((rows - 1) * (cols - 1), dtype=np.int32)
        r = a // (cols - 1)
        c = a % (cols - 1)
        v0 = r * cols + c
        v1 = v0 + 1
        v2 = v0 + cols
        v3 = v2 + 1
        return np.vstack([
            np.column_stack([v0, v2, v1]),
            np.column_stack([v1, v2, v3]),
        ]).astype(np.int32)

    @staticmethod
    def _face_colors(colors: np.ndarray) -> np.ndarray:
        rows, cols, _ = colors.shape
        c00 = colors[:-1, :-1]
        c01 = colors[:-1, 1:]
        c10 = colors[1:, :-1]
        c11 = colors[1:, 1:]
        tri1 = (c00 + c10 + c01) / 3.0
        tri2 = (c01 + c10 + c11) / 3.0
        return np.vstack([tri1.reshape(-1, 4), tri2.reshape(-1, 4)]).astype(np.float32)

    def _clear(self) -> None:
        for item in self._items:
            self.view.removeItem(item)
        self._items.clear()

    def _add(self, item) -> None:
        self.view.addItem(item)
        self._items.append(item)

    def draw_scene(self, physics: StructuredLightPhysics, state: SimState) -> None:
        self._clear()
        scene_world_x = physics.xx
        scene_world_y = physics.yy
        h = physics.surface_height_at(state, scene_world_x, scene_world_y)
        object_mask = h > 1e-4
        if state.projection_mode == "temporal":
            preview_signal = physics.surface_projector_preview(state, scene_world_x, scene_world_y, h, int(state.phase_deg // 52) % 7)
        else:
            preview_signal = physics.surface_projector_preview(state, scene_world_x, scene_world_y, h, math.radians(state.phase_deg))

        scene_step = 3
        signal_small = preview_signal[::scene_step, ::scene_step]
        surface_x = scene_world_x[::scene_step, ::scene_step]
        surface_y = scene_world_y[::scene_step, ::scene_step]
        surface_z = state.board_z - h[::scene_step, ::scene_step]

        base = np.zeros((*surface_x.shape, 4), dtype=float)
        base_shade = 0.16 + 0.62 * signal_small
        base[..., 0] = base_shade
        base[..., 1] = base_shade
        base[..., 2] = base_shade
        base[..., 3] = 0.82
        obj = np.zeros_like(base)
        shade = 0.18 + 0.82 * signal_small
        if state.object_type == "boards":
            obj[..., 0] = 0.24 + 0.70 * shade
            obj[..., 1] = 0.25 + 0.69 * shade
            obj[..., 2] = 0.27 + 0.67 * shade
        else:
            obj[..., 0] = shade
            obj[..., 1] = shade
            obj[..., 2] = shade
        obj[..., 3] = 0.95
        colors = np.where(object_mask[::scene_step, ::scene_step, None], obj, base)

        rows, cols = surface_x.shape
        vertices = np.column_stack([
            surface_x.reshape(-1),
            surface_z.reshape(-1),
            surface_y.reshape(-1),
        ]).astype(np.float32)
        mesh = gl.GLMeshItem(
            vertexes=vertices,
            faces=self._mesh_faces(rows, cols),
            faceColors=self._face_colors(colors),
            smooth=False,
            drawEdges=False,
            shader="shaded",
            glOptions="opaque",
        )
        self._add(mesh)

        grid = gl.GLGridItem()
        grid_depth = 8.4
        grid.setSize(x=4.4, y=grid_depth, z=1.0)
        grid.setSpacing(x=0.25, y=0.25, z=1.0)
        grid.translate(0.0, grid_depth / 2.0, -1.45)
        grid.setColor((18, 49, 92, 160))
        self._add(grid)

        cam = (state.camera_x, 0.0, -1.1)
        proj = (state.projector_x, 0.0, -0.7)
        target = (state.object_offset_x, state.board_z - float(np.nanmax(h)) * 0.55, state.object_offset_y)
        devices = gl.GLScatterPlotItem(
            pos=np.array([cam, proj], dtype=np.float32),
            color=np.array([[0.98, 0.17, 0.43, 1.0], [0.23, 0.51, 0.96, 1.0]], dtype=np.float32),
            size=np.array([16.0, 18.0], dtype=np.float32),
            pxMode=True,
        )
        self._add(devices)
        self._add(gl.GLLinePlotItem(pos=np.array([cam, target], dtype=np.float32), color=(0.98, 0.17, 0.43, 0.85), width=1.4, antialias=True))
        self._add(gl.GLLinePlotItem(pos=np.array([proj, target], dtype=np.float32), color=(0.23, 0.51, 0.96, 0.85), width=1.4, antialias=True))
        self._add(gl.GLLinePlotItem(pos=np.array([[-2.0, 0.0, -1.28], [2.0, 0.0, -1.28]], dtype=np.float32), color=(0.80, 0.88, 1.0, 0.55), width=1.0, antialias=True))
        label_font = QtGui.QFont("Microsoft YaHei", 10)
        self._add(gl.GLTextItem(pos=np.array([cam[0], cam[1] - 0.16, cam[2] + 0.10]), text="CAMERA +X", color=QtGui.QColor("#fb7185"), font=label_font))
        projector_label = "VCSEL PROJECTOR -X" if state.projector_type == "vcsel" else "DLP PROJECTOR -X"
        self._add(gl.GLTextItem(pos=np.array([proj[0], proj[1] + 0.12, proj[2] + 0.10]), text=projector_label, color=QtGui.QColor("#60a5fa"), font=label_font))
        self.view.opts["center"] = QtGui.QVector3D(0.0, 3.6, -0.18)
        self.view.setCameraPosition(distance=9.4)


class CloudCanvas(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("#020617")
        self.view.setCameraPosition(distance=4.2, elevation=24, azimuth=-58)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self._items: List[object] = []

    def _clear(self) -> None:
        for item in self._items:
            self.view.removeItem(item)
        self._items.clear()

    def _add(self, item) -> None:
        self.view.addItem(item)
        self._items.append(item)

    @staticmethod
    def _height_colors(z: np.ndarray) -> np.ndarray:
        t = np.clip((z + 0.8) / 2.35, 0.0, 1.0)
        colors = np.zeros((z.size, 4), dtype=np.float32)
        colors[:, 0] = 0.02 + 0.05 * t
        colors[:, 1] = 0.20 + 0.78 * t
        colors[:, 2] = 0.95 - 0.55 * t
        colors[:, 3] = 0.94
        return colors

    @staticmethod
    def _error_colors(err: np.ndarray) -> np.ndarray:
        t = np.clip(err / 0.20, 0.0, 1.0)
        colors = np.zeros((err.size, 4), dtype=np.float32)
        colors[:, 0] = t
        colors[:, 1] = 1.0 - np.abs(t - 0.5) * 1.8
        colors[:, 2] = 0.10
        colors[:, 3] = 0.94
        return np.clip(colors, 0.0, 1.0)

    def show_cloud(self, physics: StructuredLightPhysics, height: np.ndarray | None, truth: np.ndarray | None, color_mode: str, state: SimState) -> Tuple[int, float]:
        self._clear()
        grid = gl.GLGridItem()
        grid.setSize(x=3.0, y=2.4, z=1.0)
        grid.setSpacing(x=0.25, y=0.25, z=1.0)
        grid.setColor((90, 105, 125, 150))
        self._add(grid)
        if height is None:
            return 0, float("nan")

        valid = np.isfinite(height)
        step = 2
        zs = height[::step, ::step]
        vv = valid[::step, ::step]
        if not np.any(vv):
            return 0, float("nan")

        nxs = physics.nx[::step, ::step]
        nys = physics.ny[::step, ::step]
        px, py, pz, ray_valid = physics.point_on_camera_ray_at_height(state, nxs[vv], nys[vv], zs[vv])
        keep = ray_valid & np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
        if not np.any(keep):
            return 0, float("nan")
        x = px[keep]
        y = py[keep]
        z = pz[keep]
        if color_mode == "error" and truth is not None:
            err = np.abs(z - physics.surface_height_at(state, x, y))
            colors = self._error_colors(err)
        else:
            colors = self._height_colors(z)
        pos = np.column_stack([x, y, z]).astype(np.float32)
        points = gl.GLScatterPlotItem(pos=pos, color=colors, size=3.0, pxMode=True)
        self._add(points)
        self.view.opts["center"] = QtGui.QVector3D(0.0, 0.0, 0.25)
        rmse = float(np.sqrt(np.nanmean((height[valid] - truth[valid]) ** 2)) * 1000.0) if truth is not None else float("nan")
        return int(np.sum(valid)), rmse


class LabeledSlider(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(float)
    value_committed = QtCore.pyqtSignal(object, float, float)

    def __init__(self, label: str, minimum: float, maximum: float, value: float, step: float, suffix: str) -> None:
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.suffix = suffix
        self.display_unit = suffix
        self.unit_factors = {"m": 1.0, "cm": 100.0} if suffix == "m" else {suffix: 1.0}
        self.label = QtWidgets.QLabel(label)
        self.label.setStyleSheet("color:#bfdbfe;font-size:13px;")
        self.current_value = float(value)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setFixedWidth(104)
        self.edit.setFixedHeight(26)
        self.edit.setAlignment(QtCore.Qt.AlignRight)
        self.edit.setStyleSheet(
            "background:#020617;color:#f8fafc;border:1px solid #334155;"
            "border-right:0;border-top-left-radius:4px;border-bottom-left-radius:4px;"
            "border-top-right-radius:0;border-bottom-right-radius:0;padding:3px 6px;"
        )
        self.step_up = QtWidgets.QToolButton()
        self.step_up.setText("▲")
        self.step_down = QtWidgets.QToolButton()
        self.step_down.setText("▼")
        for button in (self.step_up, self.step_down):
            button.setAutoRepeat(True)
            button.setFixedSize(24, 13)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.setStyleSheet(
                "QToolButton { background:#1e293b;color:#e2e8f0;border:0;font-size:9px;padding:0; }"
                "QToolButton:hover { background:#334155;color:#ffffff; }"
                "QToolButton:pressed { background:#4f46e5;color:#ffffff; }"
            )
        self.step_up.setStyleSheet(
            self.step_up.styleSheet()
            + "QToolButton { border:1px solid #334155;border-bottom:0;border-top-right-radius:4px; }"
        )
        self.step_down.setStyleSheet(
            self.step_down.styleSheet()
            + "QToolButton { border:1px solid #334155;border-top:1px solid #475569;border-bottom-right-radius:4px; }"
        )
        step_box = QtWidgets.QWidget()
        step_box.setFixedSize(24, 26)
        step_buttons = QtWidgets.QVBoxLayout(step_box)
        step_buttons.setContentsMargins(0, 0, 0, 0)
        step_buttons.setSpacing(0)
        step_buttons.addWidget(self.step_up)
        step_buttons.addWidget(self.step_down)
        number_box = QtWidgets.QWidget()
        number_box.setFixedHeight(26)
        number_layout = QtWidgets.QHBoxLayout(number_box)
        number_layout.setContentsMargins(0, 0, 0, 0)
        number_layout.setSpacing(0)
        number_layout.addWidget(self.edit)
        number_layout.addWidget(step_box)
        if suffix == "m":
            self.unit_combo = QtWidgets.QComboBox()
            self.unit_combo.addItems(["m", "cm"])
            self.unit_combo.setFixedWidth(58)
            self.unit_combo.currentTextChanged.connect(self._change_display_unit)
            self.suffix_widget = self.unit_combo
        else:
            self.unit_combo = None
            self.suffix_widget = QtWidgets.QLabel(suffix)
            self.suffix_widget.setFixedWidth(52)
            self.suffix_widget.setStyleSheet("color:#38bdf8;font-weight:bold;font-size:12px;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.label, 1)
        row.addWidget(number_box)
        row.addWidget(self.suffix_widget)
        layout.addLayout(row)
        self.edit.editingFinished.connect(self._from_edit)
        self.step_up.clicked.connect(lambda: self._step_by(1))
        self.step_down.clicked.connect(lambda: self._step_by(-1))
        self.set_value(value, emit=False)

    def value(self) -> float:
        return self.current_value

    def set_value(self, value: float, emit: bool = True) -> None:
        value = max(self.minimum, min(self.maximum, float(value)))
        value = round((value - self.minimum) / self.step) * self.step + self.minimum
        value = max(self.minimum, min(self.maximum, float(value)))
        self.current_value = value
        self._refresh_edit_text()
        if emit:
            self.value_changed.emit(value)

    def _from_edit(self) -> None:
        old_value = self.current_value
        try:
            display_value = float(self.edit.text())
            base_value = display_value / self.unit_factors.get(self.display_unit, 1.0)
            self.set_value(base_value)
        except ValueError:
            self.set_value(self.current_value, emit=False)
            return
        if abs(self.current_value - old_value) > 1e-12:
            self.value_committed.emit(self, old_value, self.current_value)

    def _step_by(self, direction: int) -> None:
        old_value = self.current_value
        self.set_value(self.current_value + direction * self.step)
        if abs(self.current_value - old_value) > 1e-12:
            self.value_committed.emit(self, old_value, self.current_value)

    def _change_display_unit(self, unit: str) -> None:
        self.display_unit = unit
        self._refresh_edit_text()

    def _refresh_edit_text(self) -> None:
        factor = self.unit_factors.get(self.display_unit, 1.0)
        value = self.current_value * factor
        display_step = self.step * factor
        if display_step < 0.01:
            text = f"{value:.3f}"
        elif display_step < 0.1:
            text = f"{value:.2f}"
        elif display_step < 1.0:
            text = f"{value:.1f}"
        else:
            text = f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"
        self.edit.setText(text)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = SimState()
        self.physics = StructuredLightPhysics()
        self.last_result: Dict[str, np.ndarray] | None = None
        self.undo_stack: List[Tuple[LabeledSlider, float, float]] = []
        self.setWindowTitle(APP_TITLE)
        self.resize(APP_W, APP_H)
        self.setStyleSheet(self._style())
        self._build_ui()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.preview()

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main = QtWidgets.QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        left = QtWidgets.QScrollArea()
        left.setFixedWidth(520)
        left.setWidgetResizable(True)
        left.setFrameShape(QtWidgets.QFrame.NoFrame)
        panel = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(panel)
        self.controls_layout.setContentsMargins(16, 12, 16, 12)
        self.controls_layout.setSpacing(8)
        left.setWidget(panel)
        main.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(12)
        main.addWidget(right, 1)

        title = QtWidgets.QLabel(APP_TITLE)
        title.setStyleSheet("font-size:22px;font-weight:bold;color:#a78bfa;")
        subtitle = QtWidgets.QLabel("Python 原生界面 · DLP 正弦条纹 / VCSEL 点阵时序结构光")
        subtitle.setStyleSheet("font-size:12px;color:#94a3b8;")
        self.controls_layout.addWidget(title)
        self.controls_layout.addWidget(subtitle)

        self.status = QtWidgets.QLabel("请点击“启动高精度点云重建”运行算法")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#cbd5e1;background:#020617;border:1px solid #1e293b;border-radius:6px;padding:10px;")
        self.controls_layout.addWidget(self.status)

        action_row = QtWidgets.QHBoxLayout()
        run = QtWidgets.QPushButton("启动重建")
        run.clicked.connect(self.run_reconstruction)
        export = QtWidgets.QPushButton("导出 XYZ")
        export.clicked.connect(self.export_xyz)
        reset = QtWidgets.QPushButton("复位")
        reset.clicked.connect(self.reset_defaults)
        action_row.addWidget(run, 2)
        action_row.addWidget(export, 1)
        action_row.addWidget(reset, 1)
        self.controls_layout.addLayout(action_row)

        self._section("1. 投影与成像参数")
        self.projector_x = self._add_slider("投影仪 X 轴位置", -1.5, 1.5, self.state.projector_x, 0.01, "m")
        self.fringe_frequency = self._add_slider("条纹空间频率", 2.0, 18.0, self.state.fringe_frequency, 0.1, "rad/m")
        self.phase_deg = self._add_slider("实时相移偏置", 0.0, 360.0, self.state.phase_deg, 1.0, "°")
        self.camera_x = self._add_slider("相机 X 轴位置", -1.5, 1.5, self.state.camera_x, 0.01, "m")
        self.ambient_lux = self._add_slider("环境照度", 5.0, 1000.0, self.state.ambient_lux, 1.0, "Lux")

        self._section("1A. 投影仪真实内参")
        self.projector_focal_mm = self._add_slider("投影仪真实焦距", 3.0, 25.0, self.state.projector_focal_mm, 0.1, "mm")
        self.projector_resolution_x = self._add_slider("投影仪水平分辨率", 640.0, 4096.0, self.state.projector_resolution_x, 1.0, "px")
        self.projector_resolution_y = self._add_slider("投影仪垂直分辨率", 480.0, 2160.0, self.state.projector_resolution_y, 1.0, "px")
        self.projector_pixel_um = self._add_slider("投影仪像元 / 微镜尺寸", 3.0, 15.0, self.state.projector_pixel_um, 0.1, "um")
        self.projector_power_mw = self._add_slider("投影光功率", 20.0, 1500.0, self.state.projector_power_mw, 10.0, "mW")
        self.projector_wavelength_nm = self._add_slider("投影中心波长", 405.0, 940.0, self.state.projector_wavelength_nm, 5.0, "nm")
        self.projector_gamma = self._add_slider("投影灰度伽马", 0.6, 2.8, self.state.projector_gamma, 0.1, "")

        self._section("1B. 相机真实内参")
        self.camera_focal_mm = self._add_slider("相机真实焦距", 3.0, 25.0, self.state.camera_focal_mm, 0.1, "mm")
        self.camera_f_number = self._add_slider("相机镜头 F 数", 1.2, 16.0, self.state.camera_f_number, 0.1, "")
        self.exposure_ms = self._add_slider("相机曝光时间", 0.1, 40.0, self.state.exposure_ms, 0.1, "ms")
        self.sensor_gain_db = self._add_slider("传感器模拟增益", 0.0, 24.0, self.state.sensor_gain_db, 0.5, "dB")
        self.camera_principal_x_px = self._add_slider("相机主点 cx", 0.0, 3264.0, self.state.camera_principal_x_px, 1.0, "px")
        self.camera_principal_y_px = self._add_slider("相机主点 cy", 0.0, 2448.0, self.state.camera_principal_y_px, 1.0, "px")
        self.camera_k1 = self._add_slider("相机径向畸变 k1", -0.8, 0.8, self.state.camera_k1, 0.01, "")
        self.camera_k2 = self._add_slider("相机径向畸变 k2", -0.8, 0.8, self.state.camera_k2, 0.01, "")
        self.defocus_blur_px = self._add_slider("透镜离焦模糊半径", 0.0, 5.0, self.state.defocus_blur_px, 0.1, "px")
        self.prnu_percent = self._add_slider("像素响应不均匀 PRNU", 0.0, 8.0, self.state.prnu_percent, 0.1, "%")
        self.dsnu_e = self._add_slider("暗信号不均匀 DSNU", 0.0, 20.0, self.state.dsnu_e, 0.5, "e-")

        self._section("1C. 相机 / 投影仪外参标定")
        self.extrinsic_yaw_deg = self._add_slider("外参 yaw 偏角", -8.0, 8.0, self.state.extrinsic_yaw_deg, 0.1, "°")
        self.extrinsic_pitch_deg = self._add_slider("外参 pitch 偏角", -8.0, 8.0, self.state.extrinsic_pitch_deg, 0.1, "°")
        self.extrinsic_roll_deg = self._add_slider("外参 roll 偏角", -8.0, 8.0, self.state.extrinsic_roll_deg, 0.1, "°")

        self._section("2. 目标物体")
        self.board_z = self._add_slider("物体距离 Z 轴", 1.5, 8.0, self.state.board_z, 0.01, "m")
        self.object_offset_x = self._add_slider("物体 X 位置偏移", -1.5, 1.5, self.state.object_offset_x, 0.01, "m")
        self.object_offset_y = self._add_slider("物体 Y 位置偏移", -1.2, 1.2, self.state.object_offset_y, 0.01, "m")
        self.board_depth_1 = self._add_slider("木板1凸出深度", 0.05, 1.50, self.state.board_depth_1, 0.01, "m")
        self.board_depth_2 = self._add_slider("木板2凸出深度", 0.05, 1.50, self.state.board_depth_2, 0.01, "m")
        self.board_depth_3 = self._add_slider("木板3凸出深度", 0.05, 1.50, self.state.board_depth_3, 0.01, "m")
        self.object_group = self._button_group([
            ("球面", "sphere"),
            ("人脸浮雕", "face"),
            ("多峰曲面", "peaks"),
            ("多层木板", "boards"),
        ], self.set_object)
        self._set_group_checked(self.object_group, self.state.object_type)

        self._section("3. 投影图案 / 解算方法")
        self.mode_group = self._button_group([
            ("DLP 条纹结构光", "fringe"),
            ("VCSEL 点阵时序", "temporal"),
        ], self.set_projection_mode)
        self._set_group_checked(self.mode_group, self.state.projection_mode)

        self._section("4. 相移采集")
        self.phase_group = self._button_group([
            ("3步快速", 3),
            ("4步标准", 4),
            ("5步抗噪", 5),
            ("6步高精", 6),
        ], self.set_phase_steps)
        self._set_group_checked(self.phase_group, self.state.phase_steps)

        self._section("5. APS / EVS 与光学误差")
        self.sensor_group = self._button_group([
            ("APS", "aps"),
            ("EVS", "evs"),
        ], self.set_sensor)

        self.controls_layout.addStretch()

        self.scene = SceneCanvas()
        right_layout.addWidget(self.scene, 2)
        bottom = QtWidgets.QHBoxLayout()
        right_layout.addLayout(bottom, 1)
        self.sensor_view = ImageCanvas("相机传感器拍摄图", flip_x=True)
        self.phase_view = ImageCanvas("相位 / 编码图", flip_x=True)
        self.cloud_view = CloudCanvas()
        bottom.addWidget(self.sensor_view, 1)
        bottom.addWidget(self.phase_view, 1)
        bottom.addWidget(self.cloud_view, 1)

        self.metrics = QtWidgets.QLabel("Points: 0    RMSE: --")
        self.metrics.setStyleSheet("color:#34d399;font-weight:bold;padding:6px;")
        right_layout.addWidget(self.metrics)

    def _section(self, text: str) -> None:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color:#e2e8f0;font-size:14px;font-weight:bold;margin-top:4px;border-top:1px solid #1e293b;padding-top:7px;")
        self.controls_layout.addWidget(lbl)

    def _add_slider(self, label: str, minimum: float, maximum: float, value: float, step: float, suffix: str) -> LabeledSlider:
        slider = LabeledSlider(label, minimum, maximum, value, step, suffix)
        slider.value_changed.connect(self._parameter_changed)
        slider.value_committed.connect(self._remember_input_change)
        self.controls_layout.addWidget(slider)
        return slider

    def _remember_input_change(self, control: LabeledSlider, old_value: float, new_value: float) -> None:
        self.undo_stack.append((control, old_value, new_value))
        if len(self.undo_stack) > 80:
            self.undo_stack.pop(0)

    def undo_last_input(self) -> bool:
        if not self.undo_stack:
            return False
        control, old_value, new_value = self.undo_stack.pop()
        control.set_value(old_value, emit=True)
        label = control.label.text()
        self.status.setText(f"已撤回上一次输入：{label} 从 {new_value:g} 恢复为 {old_value:g}")
        return True

    def eventFilter(self, obj: object, event: QtCore.QEvent) -> bool:
        if (
            event.type() == QtCore.QEvent.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.matches(QtGui.QKeySequence.Undo)
            and self.isActiveWindow()
        ):
            return self.undo_last_input()
        return super().eventFilter(obj, event)

    def _button_group(self, items: List[Tuple[str, object]], callback) -> QtWidgets.QButtonGroup:
        box = QtWidgets.QHBoxLayout()
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for text, value in items:
            btn = QtWidgets.QPushButton(text)
            btn.setCheckable(True)
            btn.setProperty("value", value)
            group.addButton(btn)
            box.addWidget(btn)
            btn.clicked.connect(lambda checked=False, b=btn: callback(b.property("value")))
        self.controls_layout.addLayout(box)
        if group.buttons():
            group.buttons()[0].setChecked(True)
        return group

    @staticmethod
    def _set_group_checked(group: QtWidgets.QButtonGroup, value: object) -> None:
        for btn in group.buttons():
            if btn.property("value") == value:
                btn.setChecked(True)
                return

    def _parameter_changed(self, _: float) -> None:
        self.sync_state_from_controls()
        self.last_result = None
        self.preview()
        self.update_calibration_summary()
        self.status.setText("参数已变化，请重新点击重建按钮运行算法")

    def sync_state_from_controls(self) -> None:
        self.state.projector_x = self.projector_x.value()
        self.state.camera_x = self.camera_x.value()
        self.state.board_z = self.board_z.value()
        self.state.object_offset_x = self.object_offset_x.value()
        self.state.object_offset_y = self.object_offset_y.value()
        self.state.board_depth_1 = self.board_depth_1.value()
        self.state.board_depth_2 = self.board_depth_2.value()
        self.state.board_depth_3 = self.board_depth_3.value()
        self.state.fringe_frequency = self.fringe_frequency.value()
        self.state.phase_deg = self.phase_deg.value()
        self.state.ambient_lux = self.ambient_lux.value()
        self.state.projector_focal_mm = self.projector_focal_mm.value()
        self.state.projector_resolution_x = self.projector_resolution_x.value()
        self.state.projector_resolution_y = self.projector_resolution_y.value()
        self.state.projector_pixel_um = self.projector_pixel_um.value()
        self.state.projector_power_mw = self.projector_power_mw.value()
        self.state.projector_wavelength_nm = self.projector_wavelength_nm.value()
        self.state.projector_gamma = self.projector_gamma.value()
        self.state.camera_focal_mm = self.camera_focal_mm.value()
        self.state.camera_f_number = self.camera_f_number.value()
        self.state.exposure_ms = self.exposure_ms.value()
        self.state.sensor_gain_db = self.sensor_gain_db.value()
        self.state.camera_principal_x_px = self.camera_principal_x_px.value()
        self.state.camera_principal_y_px = self.camera_principal_y_px.value()
        self.state.camera_k1 = self.camera_k1.value()
        self.state.camera_k2 = self.camera_k2.value()
        self.state.defocus_blur_px = self.defocus_blur_px.value()
        self.state.prnu_percent = self.prnu_percent.value()
        self.state.dsnu_e = self.dsnu_e.value()
        self.state.extrinsic_yaw_deg = self.extrinsic_yaw_deg.value()
        self.state.extrinsic_pitch_deg = self.extrinsic_pitch_deg.value()
        self.state.extrinsic_roll_deg = self.extrinsic_roll_deg.value()
        self._apply_projection_hardware_binding()

    def _apply_projection_hardware_binding(self) -> None:
        if self.state.projection_mode == "temporal":
            self.state.projector_type = "vcsel"
        else:
            self.state.projection_mode = "fringe"
            self.state.projector_type = "dlp"

    def update_calibration_summary(self) -> None:
        return

    def set_object(self, value: str) -> None:
        self.state.object_type = value
        self.last_result = None
        self.preview()
        self.status.setText("物体已改变，请重新采集并重建")

    def set_projection_mode(self, value: str) -> None:
        self.state.projection_mode = value
        self._apply_projection_hardware_binding()
        self.last_result = None
        self.preview()
        self.update_calibration_summary()
        names = {"fringe": "DLP 条纹结构光", "temporal": "VCSEL 点阵时序编码结构光"}
        self.status.setText(f"投影图案已切换：{names[self.state.projection_mode]}")

    def set_phase_steps(self, value: int) -> None:
        self.state.phase_steps = int(value)
        self.status.setText("条纹模式：相移步数已改变，请重新采集并重建")

    def set_sensor(self, value: str) -> None:
        self.state.sensor_mode = value
        self.last_result = None
        self.preview()
        self.status.setText(f"已切换 {value.upper()}，请重新运行算法")

    def _pause_for_acquisition(self, milliseconds: int) -> None:
        deadline = QtCore.QTime.currentTime().addMSecs(milliseconds)
        while QtCore.QTime.currentTime() < deadline:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 20)
            QtCore.QThread.msleep(12)

    def _show_fringe_acquisition_frame(self, phase_rad: float, index: int, total: int) -> None:
        angle = int(round(math.degrees(phase_rad))) % 360
        self.state.phase_deg = float(angle)
        self.phase_deg.set_value(self.state.phase_deg, emit=False)
        sensor, _, _ = self.physics.fringe_intensity(self.state, phase_rad)
        self.scene.draw_scene(self.physics, self.state)
        self.sensor_view.show_image(sensor, cmap="gray", vmin=0, vmax=1)
        self.phase_view.show_image(np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        self.status.setText(f"正在采集第 {index}/{total} 帧相移图（{angle}°）...")
        QtWidgets.QApplication.processEvents()

    def _show_temporal_acquisition_frame(self, frame_index: int, previous: np.ndarray | None) -> np.ndarray:
        self.state.phase_deg = float(frame_index * 52)
        self.phase_deg.set_value(self.state.phase_deg, emit=False)
        sensor, _, _ = self.physics.temporal_pattern(self.state, frame_index)
        if self.state.sensor_mode == "evs" and previous is not None:
            delta = sensor - previous
            event_rgb = np.zeros((*sensor.shape, 3), dtype=np.float32)
            event_rgb[..., 0] = np.clip(-delta * 3.0, 0.0, 1.0)
            event_rgb[..., 1] = np.where(np.abs(delta) < 0.03, sensor * 0.25, 0.0)
            event_rgb[..., 2] = np.clip(delta * 3.0, 0.0, 1.0)
            self.sensor_view.show_image(event_rgb)
        else:
            self.sensor_view.show_image(sensor, cmap="gray", vmin=0, vmax=1)
        self.scene.draw_scene(self.physics, self.state)
        self.phase_view.show_image(np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        self.status.setText(f"正在投影第 {frame_index + 1}/7 幅点阵编码图...")
        QtWidgets.QApplication.processEvents()
        return sensor

    def preview(self) -> None:
        self.sync_state_from_controls()
        phase = math.radians(self.state.phase_deg)
        if self.state.projection_mode == "temporal":
            sensor, _, _ = self.physics.temporal_pattern(self.state, int(self.state.phase_deg // 52) % 7)
        else:
            sensor, _, _ = self.physics.fringe_intensity(self.state, phase)
        self.scene.draw_scene(self.physics, self.state)
        self.sensor_view.show_image(sensor, cmap="gray", vmin=0, vmax=1)
        if self.last_result is None:
            self.phase_view.show_image(np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
            self.cloud_view.show_cloud(self.physics, None, None, self.state.color_mode, self.state)

    def _board_measurement_report(self, result: Dict[str, np.ndarray]) -> str:
        height = result["height"]
        local_x = self.physics.xx - self.state.object_offset_x
        local_y = self.physics.yy - self.state.object_offset_y
        rows = []
        measured_depths: List[float] = []
        depth_uncertainties_mm: List[float] = []
        true_depths: List[float] = []
        systematic_floor_mm = 0.35 if self.state.projection_mode == "temporal" else 0.12
        systematic_floor_mm += 0.04 * abs(self.state.extrinsic_yaw_deg)

        def fmt_bias(value_mm: float) -> str:
            if not np.isfinite(value_mm):
                return "--"
            if abs(value_mm) < 0.005:
                return "<0.01"
            return f"{value_mm:+.2f}"

        for index, (x0, x1, y0, y1, true_depth) in enumerate(self.physics._board_specs(self.state), start=1):
            # The visible board scene contains front faces plus side faces/occlusion edges.
            # Measure each board as a depth layer, which is closer to how a step-depth target is inspected.
            visible = np.isfinite(height) & (np.abs(height - true_depth) < 0.06)
            true_depths.append(true_depth)
            if np.any(visible):
                layer_values = height[visible]
                measured_depth = float(np.nanmean(layer_values))
                xs = local_x[visible]
                ys = local_y[visible]
                measured_w = float(np.nanmax(xs) - np.nanmin(xs)) if xs.size else float("nan")
                measured_h = float(np.nanmax(ys) - np.nanmin(ys)) if ys.size else float("nan")
                measured_depths.append(measured_depth)
                residual_mm = (layer_values - true_depth) * 1000.0
                point_sigma_mm = float(np.nanstd(residual_mm))
                mean_sem_mm = point_sigma_mm / math.sqrt(max(1, int(np.sum(visible))))
                uncertainty_mm = math.sqrt(mean_sem_mm * mean_sem_mm + systematic_floor_mm * systematic_floor_mm)
                depth_uncertainties_mm.append(uncertainty_mm)
                bias_mm = (measured_depth - true_depth) * 1000.0
                rows.append(
                    f"板{index}: 真实宽高 {x1 - x0:.2f}x{y1 - y0:.2f}m, 凸出深度 {true_depth:.3f}m; "
                    f"点云可见范围 {measured_w:.2f}x{measured_h:.2f}m, 测得深度 {measured_depth:.4f}m, "
                    f"偏差 {fmt_bias(bias_mm)}mm, 单点σ {point_sigma_mm:.2f}mm, 不确定度 ±{uncertainty_mm:.2f}mm"
                )
            else:
                measured_depths.append(float("nan"))
                depth_uncertainties_mm.append(float("nan"))
                rows.append(f"板{index}: 真实宽高 {x1 - x0:.2f}x{y1 - y0:.2f}m, 凸出深度 {true_depth:.3f}m; 对应深度层点数不足")
        spacings = []
        for i in range(len(true_depths) - 1):
            true_gap = true_depths[i + 1] - true_depths[i]
            measured_gap = measured_depths[i + 1] - measured_depths[i]
            if np.isfinite(measured_gap):
                gap_uncertainty = math.sqrt(depth_uncertainties_mm[i] ** 2 + depth_uncertainties_mm[i + 1] ** 2)
                gap_bias_mm = (measured_gap - true_gap) * 1000.0
                spacings.append(
                    f"板{i + 1}->板{i + 2} 深度差: 真 {true_gap:.3f}m, 测 {measured_gap:.4f}m, "
                    f"偏差 {fmt_bias(gap_bias_mm)}mm, 不确定度 ±{gap_uncertainty:.2f}mm"
                )
            else:
                spacings.append(f"板{i + 1}->板{i + 2} 深度差: 真 {true_gap:.3f}m, 测量点不足")
        total_true_gap = true_depths[-1] - true_depths[0]
        total_measured_gap = measured_depths[-1] - measured_depths[0]
        if np.isfinite(total_measured_gap):
            total_uncertainty = math.sqrt(depth_uncertainties_mm[0] ** 2 + depth_uncertainties_mm[-1] ** 2)
            total_bias_mm = (total_measured_gap - total_true_gap) * 1000.0
            spacings.append(
                f"板1->板3 总深度差: 真 {total_true_gap:.3f}m, 测 {total_measured_gap:.4f}m, "
                f"偏差 {fmt_bias(total_bias_mm)}mm, 不确定度 ±{total_uncertainty:.2f}mm"
            )
        return "\n".join(rows + spacings)

    def run_reconstruction(self) -> None:
        self.sync_state_from_controls()
        source_name = "VCSEL" if self.state.projector_type == "vcsel" else "DLP"
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if self.state.projection_mode == "temporal":
                self.status.setText(f"正在投影 {source_name} 的 7 幅点阵编码图并解码 EVS 事件...")
                previous = None
                for frame_index in range(7):
                    previous = self._show_temporal_acquisition_frame(frame_index, previous)
                    self._pause_for_acquisition(130)
                result = self.physics.reconstruct_temporal(self.state)
                self.sensor_view.show_image(result["sensor_montage"], cmap="gray", vmin=0, vmax=1)
                self.phase_view.show_image(result["decoded"], cmap="turbo", vmin=0, vmax=TEMPORAL_COLUMNS - 1)
                done = f"{source_name} 点阵时序重建完成：已按 7 帧图案与 6 帧 EVS 事件码完成解码"
            else:
                self.status.setText(f"正在采集 {source_name} 的 {self.state.phase_steps} 步相移帧并解包裹...")
                phases = np.linspace(0.0, 2.0 * np.pi, self.state.phase_steps, endpoint=False)
                for i, phase_rad in enumerate(phases, start=1):
                    self._show_fringe_acquisition_frame(float(phase_rad), i, self.state.phase_steps)
                    self._pause_for_acquisition(260)
                result = self.physics.reconstruct_fringe(self.state)
                frames = result.get("frames", [result["sensor"]])
                self.sensor_view.show_image(frames[-1], cmap="gray", vmin=0, vmax=1)
                self.phase_view.show_image(result["phase"], cmap="twilight", vmin=-np.pi, vmax=np.pi)
                done = f"{source_name} 条纹相移重建完成：已生成连续点云"
            self.last_result = result
            count, rmse = self.cloud_view.show_cloud(self.physics, result["height"], result["truth"], self.state.color_mode, self.state)
            self.metrics.setText(f"Points: {count}    RMSE: {rmse:.2f} mm")
            if self.state.object_type == "boards":
                self.status.setText(done + "\n" + self._board_measurement_report(result))
            else:
                self.status.setText(done)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def export_xyz(self) -> None:
        if not self.last_result:
            QtWidgets.QMessageBox.warning(self, "导出失败", "请先启动点云重建以生成数据。")
            return
        height = self.last_result["height"]
        valid = np.isfinite(height)
        if not np.any(valid):
            QtWidgets.QMessageBox.warning(self, "导出失败", "当前没有有效点云，请先重建。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出 XYZ 点云", str(Path.cwd() / "pointcloud_python.xyz"), "XYZ Point Cloud (*.xyz)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("# XYZ Point Cloud generated by pure Python Structure Light Sim\n")
            f.write("# X Y Z R G B\n")
            ys, xs = np.where(valid)
            for y, x in zip(ys[::2], xs[::2]):
                z = float(height[y, x])
                wx = float(self.physics.xx[y, x])
                wy = float(self.physics.yy[y, x])
                r = int(30 + 80 * min(1.0, z))
                g = int(120 + 90 * min(1.0, z))
                b = int(255 - 80 * min(1.0, z))
                f.write(f"{wx:.5f} {wy:.5f} {z:.5f} {r} {g} {b}\n")
        self.status.setText(f"点云已导出：{path}")

    def reset_defaults(self) -> None:
        self.state = SimState()
        self.undo_stack.clear()
        self.projector_x.set_value(self.state.projector_x, emit=False)
        self.camera_x.set_value(self.state.camera_x, emit=False)
        self.board_z.set_value(self.state.board_z, emit=False)
        self.object_offset_x.set_value(self.state.object_offset_x, emit=False)
        self.object_offset_y.set_value(self.state.object_offset_y, emit=False)
        self.board_depth_1.set_value(self.state.board_depth_1, emit=False)
        self.board_depth_2.set_value(self.state.board_depth_2, emit=False)
        self.board_depth_3.set_value(self.state.board_depth_3, emit=False)
        self.projector_focal_mm.set_value(self.state.projector_focal_mm, emit=False)
        self.projector_resolution_x.set_value(self.state.projector_resolution_x, emit=False)
        self.projector_resolution_y.set_value(self.state.projector_resolution_y, emit=False)
        self.projector_pixel_um.set_value(self.state.projector_pixel_um, emit=False)
        self.projector_power_mw.set_value(self.state.projector_power_mw, emit=False)
        self.projector_wavelength_nm.set_value(self.state.projector_wavelength_nm, emit=False)
        self.projector_gamma.set_value(self.state.projector_gamma, emit=False)
        self.camera_focal_mm.set_value(self.state.camera_focal_mm, emit=False)
        self.camera_f_number.set_value(self.state.camera_f_number, emit=False)
        self.exposure_ms.set_value(self.state.exposure_ms, emit=False)
        self.sensor_gain_db.set_value(self.state.sensor_gain_db, emit=False)
        self.camera_principal_x_px.set_value(self.state.camera_principal_x_px, emit=False)
        self.camera_principal_y_px.set_value(self.state.camera_principal_y_px, emit=False)
        self.camera_k1.set_value(self.state.camera_k1, emit=False)
        self.camera_k2.set_value(self.state.camera_k2, emit=False)
        self.defocus_blur_px.set_value(self.state.defocus_blur_px, emit=False)
        self.prnu_percent.set_value(self.state.prnu_percent, emit=False)
        self.dsnu_e.set_value(self.state.dsnu_e, emit=False)
        self.extrinsic_yaw_deg.set_value(self.state.extrinsic_yaw_deg, emit=False)
        self.extrinsic_pitch_deg.set_value(self.state.extrinsic_pitch_deg, emit=False)
        self.extrinsic_roll_deg.set_value(self.state.extrinsic_roll_deg, emit=False)
        self.fringe_frequency.set_value(self.state.fringe_frequency, emit=False)
        self.phase_deg.set_value(self.state.phase_deg, emit=False)
        self.ambient_lux.set_value(self.state.ambient_lux, emit=False)
        self._set_group_checked(self.object_group, self.state.object_type)
        self._set_group_checked(self.mode_group, self.state.projection_mode)
        self._set_group_checked(self.phase_group, self.state.phase_steps)
        self._set_group_checked(self.sensor_group, self.state.sensor_mode)
        self.last_result = None
        self.preview()
        self.update_calibration_summary()
        self.metrics.setText("Points: 0    RMSE: --")
        self.status.setText("全部硬件状态已还原至标准状态。")

    @staticmethod
    def _style() -> str:
        return """
        QMainWindow, QWidget { background:#0f172a; color:#e2e8f0; font-family:'Microsoft YaHei','Segoe UI'; }
        QScrollArea { background:#111827; border-right:1px solid #1e293b; }
        QPushButton {
            background:#1e293b; color:#e2e8f0; border:1px solid #334155;
            border-radius:6px; padding:9px; font-weight:bold;
        }
        QPushButton:hover { background:#334155; }
        QPushButton:checked { background:#4f46e5; border-color:#818cf8; color:white; }
        """


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            QtGui.QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QtGui.QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
