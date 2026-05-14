from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

import pyglet
from pyglet.gl import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_TEXTURE_2D,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_CLAMP_TO_EDGE,
    glBindTexture,
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glTexParameteri,
    glViewport,
)
from pyglet.graphics.shader import Shader, ShaderProgram


@dataclass
class Sample:
    name: str
    ptt: Path
    jpg: Path | None
    ac_jpg: Path | None


@dataclass
class MeshData:
    sample: Sample
    width: int
    height: int
    vertices: np.ndarray
    normals: np.ndarray
    texcoords: np.ndarray
    indices: np.ndarray
    texture_path: Path
    z_scale: float
    z95: float
    header_uv_x: float
    low_clip_raw: float
    board_ref_raw: float


def choose_texture(sample: Sample, use_ac: bool) -> Path:
    texture_path = sample.ac_jpg if use_ac and sample.ac_jpg else sample.jpg or sample.ac_jpg
    if texture_path is None:
        raise ValueError(f"{sample.name}: no JPG texture found")
    return texture_path


def find_samples(paths: list[Path]) -> list[Sample]:
    ptts: list[Path] = []
    for path in paths:
        if path.is_dir():
            ptts.extend(sorted(path.glob("*.ptt")))
        elif path.suffix.lower() == ".ptt":
            ptts.append(path)

    out: list[Sample] = []
    seen: set[Path] = set()
    for ptt in ptts:
        ptt = ptt.resolve()
        if ptt in seen:
            continue
        seen.add(ptt)
        stem = ptt.with_suffix("")
        jpg = stem.with_suffix(".jpg")
        ac = Path(str(stem) + "_AC.jpg")
        out.append(Sample(ptt.stem, ptt, jpg if jpg.exists() else None, ac if ac.exists() else None))
    return sorted(out, key=lambda sample: sample.name)


def read_ptt(path: Path) -> tuple[int, int, float, float, np.ndarray]:
    data = path.read_bytes()
    height, width = struct.unpack_from("<II", data, 0)
    pitch_x, pitch_y = struct.unpack_from("<ff", data, 8)
    expected = width * height * 3 * 2
    if len(data) - 76 != expected:
        raise ValueError(f"{path.name}: unexpected payload size")
    planes = np.frombuffer(data, dtype="<u2", offset=76).reshape(3, height, width)
    return width, height, pitch_x, pitch_y, planes


def read_header_uv_x(path: Path, width: int) -> float:
    data = path.read_bytes()
    # Bytes 32..75 are 11 packed signed int16 pairs. The first value tracks the
    # horizontal texture/depth registration offset. For 1@206 it is around -23 px,
    # matching the user's saved uv_x ~= +0.044.
    x_offsets = [struct.unpack_from("<h", data, offset)[0] for offset in range(32, 76, 4)]
    if not x_offsets:
        return 0.0
    return -float(np.median(np.asarray(x_offsets, dtype=np.float32))) / max(1, width)


def normalize_depth_preview(depth: np.ndarray, mask: np.ndarray, invert: bool) -> np.ndarray:
    valid = depth[mask].astype(np.float32)
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        high = low + 1.0
    if invert:
        normalized = (high - depth.astype(np.float32)) / (high - low)
    else:
        normalized = (depth.astype(np.float32) - low) / (high - low)
    out = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    out[~mask] = 0
    return out


def colorize_depth(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((g * 1.7 - 0.45) * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((1.0 - np.abs(g - 0.52) / 0.52) * 235, 0, 235).astype(np.uint8)
    rgb[..., 2] = np.clip((1.15 - g * 1.45) * 255, 0, 255).astype(np.uint8)
    return rgb


def export_depth_files(sample: Sample, output_dir: Path, invert: bool) -> list[Path]:
    _width, _height, _pitch_x, _pitch_y, planes = read_ptt(sample.ptt)
    depth = planes[0].copy()
    mask = depth < 60000
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "inverted" if invert else "raw"
    raw16 = depth.astype(np.uint16)
    raw16[~mask] = 0
    gray = normalize_depth_preview(depth, mask, invert)
    outputs = [
        output_dir / f"{sample.name}_plane0_raw16.png",
        output_dir / f"{sample.name}_plane0_{suffix}_preview.png",
        output_dir / f"{sample.name}_plane0_{suffix}_color.png",
        output_dir / f"{sample.name}_plane0_raw.npy",
    ]
    Image.fromarray(raw16).save(outputs[0])
    Image.fromarray(gray).save(outputs[1])
    Image.fromarray(colorize_depth(gray)).save(outputs[2])
    np.save(outputs[3], depth)
    return outputs


def resize_float(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    fill = float(np.nanmedian(array[np.isfinite(array)]))
    safe = np.nan_to_num(array, nan=fill).astype(np.float32)
    return np.asarray(Image.fromarray(safe, mode="F").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def fill_invalid_height(z_world: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = z_world.copy()
    filled[~mask] = 0.0
    valid = mask.astype(np.float32)
    # Small iterative neighbor fill. Invalid border/background becomes nearby
    # surface/base instead of punched-out holes in the texture mesh.
    for _ in range(16):
        missing = ~np.isfinite(filled) if np.isnan(filled).any() else ~mask
        if not missing.any():
            break
        padded_values = np.pad(filled, ((1, 1), (1, 1)), mode="edge")
        padded_valid = np.pad(valid, ((1, 1), (1, 1)), mode="edge")
        value_sum = (
            padded_values[:-2, 1:-1] * padded_valid[:-2, 1:-1]
            + padded_values[2:, 1:-1] * padded_valid[2:, 1:-1]
            + padded_values[1:-1, :-2] * padded_valid[1:-1, :-2]
            + padded_values[1:-1, 2:] * padded_valid[1:-1, 2:]
        )
        count = (
            padded_valid[:-2, 1:-1]
            + padded_valid[2:, 1:-1]
            + padded_valid[1:-1, :-2]
            + padded_valid[1:-1, 2:]
        )
        can_fill = (~mask) & (count > 0)
        filled[can_fill] = value_sum[can_fill] / count[can_fill]
        valid[can_fill] = 1.0
        mask = mask | can_fill
    filled[~np.isfinite(filled)] = 0.0
    return filled


def estimate_board_reference(valid: np.ndarray) -> float:
    subset = valid[valid <= np.percentile(valid, 25)]
    if subset.size < 64:
        subset = valid
    hist, edges = np.histogram(subset, bins=64)
    idx = int(hist.argmax())
    return float((edges[idx] + edges[idx + 1]) * 0.5)


def build_board_mask(texture_path: Path, size: tuple[int, int], strict: bool = False) -> np.ndarray:
    rgb = np.asarray(Image.open(texture_path).convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    if strict:
        return (g > r * 1.18) & (g > b * 1.18) & (g > 40.0) & (((r + b) * 0.5) < g * 0.78)
    return (g > r * 1.08) & (g > b * 1.08) & (g > 32.0)


def estimate_board_reference_from_texture(raw: np.ndarray, mask: np.ndarray, texture_path: Path) -> float:
    board_mask = build_board_mask(texture_path, (raw.shape[1], raw.shape[0]), strict=True)
    candidates = board_mask & mask
    if candidates.any():
        vals = raw[candidates]
        vals = vals[vals <= np.percentile(raw[mask], 40)]
        if vals.size >= 64:
            hist, edges = np.histogram(vals, bins=64)
            idx = int(hist.argmax())
            return float((edges[idx] + edges[idx + 1]) * 0.5)
    return estimate_board_reference(raw[mask])


def estimate_board_cap_from_texture(raw: np.ndarray, mask: np.ndarray, texture_path: Path) -> float | None:
    board_mask = build_board_mask(texture_path, (raw.shape[1], raw.shape[0]), strict=True)
    candidates = board_mask & mask
    if candidates.sum() < 64:
        return None
    vals = raw[candidates]
    return float(np.percentile(vals, 95))


def resize_bool(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8) * 255)
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def median_filter2d(array: np.ndarray, kernel: int = 3) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(array, pad, mode="edge")
    windows = sliding_window_view(padded, (kernel, kernel))
    return np.median(windows, axis=(-2, -1)).astype(np.float32)


def build_mesh(
    sample: Sample,
    grid: int,
    visual_z: float,
    use_ac: bool = False,
    invert_z: bool = True,
    flip_x: bool = True,
    smooth_passes: int = 1,
) -> MeshData:
    width, height, pitch_x, pitch_y, planes = read_ptt(sample.ptt)
    raw = planes[0].astype(np.float32)
    mask = raw < 60000
    valid = raw[mask]
    low_clip = float(np.percentile(valid, 10))
    high = float(np.percentile(valid, 99.5))

    scale = min(1.0, grid / max(width, height))
    mesh_size = (max(16, int(width * scale)), max(16, int(height * scale)))
    raw_filled = raw.copy()
    raw_filled[~mask] = float(np.median(valid))
    raw_r = resize_float(raw_filled, mesh_size)
    mask_r = resize_bool(mask, mesh_size)

    texture_path = choose_texture(sample, use_ac)
    board_ref = estimate_board_reference_from_texture(raw, mask, sample.jpg or texture_path)
    board_cap = estimate_board_cap_from_texture(raw, mask, sample.jpg or texture_path)

    if invert_z:
        # After texture/height alignment was corrected, the board pixels in the
        # JPG consistently map to the low-value cluster in plane0. Treat that
        # cluster as the board baseline and lift higher raw values above it.
        if board_cap is not None:
            strict_board_r = build_board_mask(sample.jpg or texture_path, mesh_size, strict=True) & mask_r
            loose_board_r = build_board_mask(sample.jpg or texture_path, mesh_size, strict=False) & mask_r
            raw_r = raw_r.copy()
            raw_r[strict_board_r] = np.minimum(raw_r[strict_board_r], board_cap)
            raw_r[loose_board_r] = np.minimum(raw_r[loose_board_r], board_cap)
        raw_delta = np.clip(raw_r - board_ref, 0.0, None)
        raw_delta[~mask_r] = 0.0
    else:
        baseline = float(np.percentile(valid, 20))
        raw_delta = np.clip(raw_r - baseline, 0.0, max(1.0, high - baseline))
        board_ref = baseline

    z_world = (raw_delta / max(1e-6, (pitch_x + pitch_y) * 0.5)).astype(np.float32)
    z_world = fill_invalid_height(z_world, mask_r)
    for _ in range(max(0, smooth_passes)):
        z_world = median_filter2d(z_world, 3)
    # In the viewer camera convention, negative Z protrudes toward the user.
    # Keep the board baseline at 0 and flip only the extrusion direction.
    z = -z_world * scale * visual_z
    h, w = z.shape

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = xx - (w - 1) * 0.5
    if flip_x:
        x = -x
    y = (h - 1) * 0.5 - yy
    pos = np.dstack([x, y, z]).reshape(-1, 3).astype(np.float32)

    filled = np.nan_to_num(z, nan=float(np.nanmedian(z[np.isfinite(z)])))
    gy, gx = np.gradient(filled)
    normals = np.dstack([-gx, -gy, np.ones_like(filled)])
    if flip_x:
        normals[..., 0] *= -1.0
    normals /= np.linalg.norm(normals, axis=2, keepdims=True).clip(1e-6)
    normals = normals.reshape(-1, 3).astype(np.float32)

    # The resized height grid samples pixel centers, not image borders.
    # Use center-based UVs so the texture camera and height samples share the
    # same sampling convention. Border-based x/(w-1) causes a visible offset.
    u = (xx + 0.5) / max(1, w)
    if flip_x:
        u = 1.0 - u
    v = 1.0 - (yy + 0.5) / max(1, h)
    uv = np.dstack([u, v]).reshape(-1, 2).astype(np.float32)

    indices: list[int] = []
    for row in range(h - 1):
        for col in range(w - 1):
            a = row * w + col
            b = a + 1
            c = a + w
            d = c + 1
            indices.extend([a, c, b, b, c, d])

    return MeshData(
        sample,
        width,
        height,
        pos,
        normals,
        uv,
        np.asarray(indices, dtype=np.uint32),
        texture_path,
        visual_z,
        float(np.percentile(z, 95)),
        read_header_uv_x(sample.ptt, width),
        low_clip,
        board_ref,
    )


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b).astype(np.float32)


def perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy * 0.5)
    out = np.zeros((4, 4), dtype=np.float32)
    out[0, 0] = f / aspect
    out[1, 1] = f
    out[2, 2] = (far + near) / (near - far)
    out[2, 3] = -1.0
    out[3, 2] = (2.0 * far * near) / (near - far)
    return out


def translate(z: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[3, 2] = z
    return out


def rotate_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]], dtype=np.float32)


def rotate_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)


VERTEX_SHADER = """
#version 330 core
in vec3 position;
in vec3 normal;
in vec2 texcoord;
uniform float yaw;
uniform float tilt;
uniform float zoom;
uniform float sceneScale;
uniform float aspect;
out vec3 v_normal;
out vec2 v_uv;
void main()
{
    v_uv = texcoord;
    float cy = cos(yaw);
    float sy = sin(yaw);
    float ct = cos(tilt);
    float st = sin(tilt);
    vec3 p = vec3(position.x * cy - position.y * sy,
                  position.x * sy + position.y * cy,
                  position.z);
    p = vec3(p.x,
             p.y * ct - p.z * st,
             p.y * st + p.z * ct);
    vec3 n = vec3(normal.x * cy - normal.y * sy,
                  normal.x * sy + normal.y * cy,
                  normal.z);
    n = vec3(n.x,
             n.y * ct - n.z * st,
             n.y * st + n.z * ct);
    v_normal = n;
    gl_Position = vec4(p.x * sceneScale * zoom / aspect,
                       p.y * sceneScale * zoom,
                       p.z * sceneScale * 0.65,
                       1.0);
}
"""


FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec2 v_uv;
uniform sampler2D tex0;
uniform float specStrength;
uniform float bumpStrength;
uniform float lightYaw;
uniform vec2 uvOffset;
out vec4 fragColor;
void main()
{
    vec2 uv = clamp(v_uv + uvOffset, vec2(0.0), vec2(1.0));
    vec3 base = texture(tex0, uv).rgb;
    vec3 n = normalize(v_normal);
    ivec2 ts = textureSize(tex0, 0);
    vec2 duv = 1.0 / vec2(max(ts.x, 1), max(ts.y, 1));
    float lumL = dot(texture(tex0, clamp(uv - vec2(duv.x, 0.0), vec2(0.0), vec2(1.0))).rgb, vec3(0.299, 0.587, 0.114));
    float lumR = dot(texture(tex0, clamp(uv + vec2(duv.x, 0.0), vec2(0.0), vec2(1.0))).rgb, vec3(0.299, 0.587, 0.114));
    float lumD = dot(texture(tex0, clamp(uv - vec2(0.0, duv.y), vec2(0.0), vec2(1.0))).rgb, vec3(0.299, 0.587, 0.114));
    float lumU = dot(texture(tex0, clamp(uv + vec2(0.0, duv.y), vec2(0.0), vec2(1.0))).rgb, vec3(0.299, 0.587, 0.114));
    vec3 detailNormal = normalize(vec3((lumL - lumR) * bumpStrength, (lumD - lumU) * bumpStrength, 1.0));
    n = normalize(n + detailNormal * 0.55);
    float cy = cos(lightYaw);
    float sy = sin(lightYaw);
    vec3 light1 = normalize(vec3(0.08 * cy, 0.08 * sy, 0.995));
    vec3 light2 = normalize(vec3(0.40, -0.28, 0.62));
    float diff = max(dot(n, light1), 0.0) * 0.68 + max(dot(n, light2), 0.0) * 0.10;
    float ambient = 0.40;
    vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0));
    vec3 halfDir = normalize(light1 + viewDir);
    float luminance = dot(base, vec3(0.299, 0.587, 0.114));
    float metalMask = smoothstep(0.50, 0.92, luminance);
    float blackPlastic = 1.0 - smoothstep(0.03, 0.20, luminance);
    float specPower = mix(34.0, 92.0, metalMask);
    float spec = pow(max(dot(n, halfDir), 0.0), specPower) * specStrength * (0.28 + metalMask * 1.45 + blackPlastic * 0.22);
    vec3 color = base * (ambient + diff) + vec3(spec);
    fragColor = vec4(color, 1.0);
}
"""


class GLViewer(pyglet.window.Window):
    def __init__(self, samples: list[Sample], grid: int, visual_z: float, uv_offset: tuple[float | None, float]) -> None:
        super().__init__(1280, 900, "Bentron AOI OpenGL 3D Viewer", resizable=True)
        self.samples = samples
        self.grid = grid
        self.visual_z = visual_z
        self.index = 0
        self.use_ac = False
        self.invert_z = True
        # Default is the orientation that matches the user-verified X-toggle alignment.
        self.flip_x = False
        self.yaw = -35.0
        self.tilt = 58.0
        self.zoom = 1.0
        self.spec_strength = 0.42
        self.bump_strength = 4.5
        self.light_yaw = 0.0
        self.smooth_passes = 1
        self.cli_uv_x = uv_offset[0]
        self.uv_offset = [0.0, uv_offset[1]]
        self.default_uv = [0.0, uv_offset[1]]
        self.last: tuple[int, int] | None = None
        self.help_visible = True
        self.program = ShaderProgram(Shader(VERTEX_SHADER, "vertex"), Shader(FRAGMENT_SHADER, "fragment"))
        self.batch = pyglet.graphics.Batch()
        self.mesh: MeshData | None = None
        self.vertex_list = None
        self.texture = None
        self.status_label = pyglet.text.Label(
            "",
            x=16,
            y=self.height - 16,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            width=max(420, self.width - 420),
            font_name="Consolas",
            font_size=15,
            color=(238, 244, 248, 255),
        )
        self.help_label = pyglet.text.Label(
            "",
            x=self.width - 16,
            y=self.height - 16,
            anchor_x="right",
            anchor_y="top",
            multiline=True,
            width=340,
            font_name="Consolas",
            font_size=15,
            color=(220, 230, 238, 255),
        )
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.load_current()

    def load_current(self) -> None:
        self.mesh = build_mesh(self.samples[self.index], self.grid, self.visual_z, self.use_ac, self.invert_z, self.flip_x, self.smooth_passes)
        self.default_uv = [(23.0 / self.mesh.width) if self.cli_uv_x is None else self.cli_uv_x, self.uv_offset[1]]
        self.uv_offset[0] = self.default_uv[0]
        image = pyglet.image.load(str(self.mesh.texture_path))
        self.texture = image.get_texture()
        glBindTexture(GL_TEXTURE_2D, self.texture.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self.batch = pyglet.graphics.Batch()
        self.vertex_list = self.program.vertex_list_indexed(
            len(self.mesh.vertices),
            GL_TRIANGLES,
            self.mesh.indices.tolist(),
            batch=self.batch,
            position=("f", self.mesh.vertices.reshape(-1).tolist()),
            normal=("f", self.mesh.normals.reshape(-1).tolist()),
            texcoord=("f", self.mesh.texcoords.reshape(-1).tolist()),
        )
        self.update_label()

    def open_file_dialog(self) -> None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Open Bentron/Pemtron PTT file",
            filetypes=[("PTT 3D files", "*.ptt"), ("All files", "*.*")],
        )
        root.destroy()
        if not selected:
            return

        selected_path = Path(selected).resolve()
        samples = find_samples([selected_path.parent])
        if not samples:
            self.status_label.text = f"No .ptt samples found in {selected_path.parent}"
            return

        self.samples = samples
        self.index = 0
        for idx, sample in enumerate(samples):
            if sample.ptt.resolve() == selected_path:
                self.index = idx
                break
        self.cli_uv_x = None
        self.use_ac = False
        self.load_current()

    def update_label(self) -> None:
        assert self.mesh is not None
        tex_name = self.mesh.texture_path.name
        self.status_label.text = (
            f"{self.mesh.sample.name}   {self.mesh.width}x{self.mesh.height}   texture={tex_name}\n"
            f"yaw={self.yaw:.1f}  tilt={self.tilt:.1f}  zoom={self.zoom:.2f}  z={self.visual_z:.2f}  "
            f"flipX={'on' if self.flip_x else 'off'}  smooth={self.smooth_passes}  p95={self.mesh.z95:.1f}\n"
            f"spec={self.spec_strength:.2f}  bump={self.bump_strength:.1f}  "
            f"uv=({self.uv_offset[0]:+.3f},{self.uv_offset[1]:+.3f})  "
            f"hdrX={self.mesh.header_uv_x:+.3f}  lowClip={self.mesh.low_clip_raw:.1f}  boardRef={self.mesh.board_ref_raw:.1f}"
        )
        self.status_label.y = self.height - 16
        self.status_label.width = max(420, self.width - 420)
        self.help_label.x = self.width - 16
        self.help_label.y = self.height - 16
        self.help_label.text = (
            "Controls\n"
            "Drag           rotate freely\n"
            "Wheel          zoom\n"
            "+ / -          height scale\n"
            "A D / Left Right   texture left right\n"
            "W S / Up Down      texture up down\n"
            "U              reset texture offset\n"
            "N / P          next previous sample\n"
            "O              open file\n"
            "T              switch texture\n"
            "X              flip left right\n"
            "H              specular\n"
            "B              bump detail\n"
            "M              mesh smoothing\n"
            "L              light direction\n"
            "C              save alignment\n"
            "E              export depth\n"
            "R              reset view\n"
            "F1             toggle help\n"
            "Esc            close"
        )

    def on_draw(self) -> None:
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        assert self.mesh is not None and self.texture is not None
        glViewport(0, 0, self.width, self.height)
        aspect = self.width / max(1, self.height)
        max_dim = max(np.ptp(self.mesh.vertices[:, 0]), np.ptp(self.mesh.vertices[:, 1]), np.ptp(self.mesh.vertices[:, 2]))
        scene_scale = 1.55 / max(1.0, max_dim)
        self.program.use()
        self.program["yaw"] = math.radians(self.yaw)
        self.program["tilt"] = math.radians(self.tilt)
        self.program["zoom"] = self.zoom
        self.program["sceneScale"] = scene_scale
        self.program["aspect"] = aspect
        self.program["specStrength"] = self.spec_strength
        self.program["bumpStrength"] = self.bump_strength
        self.program["lightYaw"] = math.radians(self.light_yaw)
        self.program["uvOffset"] = tuple(self.uv_offset)
        self.program["tex0"] = 0
        glBindTexture(GL_TEXTURE_2D, self.texture.id)
        self.batch.draw()
        glDisable(GL_DEPTH_TEST)
        self.status_label.draw()
        if self.help_visible:
            self.help_label.draw()
        glEnable(GL_DEPTH_TEST)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        self.last = (x, y)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        self.last = None

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        self.yaw += dx * 0.35
        self.tilt -= dy * 0.25
        self.update_label()

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.update_label()

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        self.zoom = min(4.0, max(0.25, self.zoom * (1.1 if scroll_y > 0 else 0.9)))

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        key = pyglet.window.key
        if symbol in (key.PLUS, key.EQUAL):
            self.visual_z *= 1.15
            self.load_current()
        elif symbol in (key.MINUS, key.UNDERSCORE):
            self.visual_z /= 1.15
            self.load_current()
        elif symbol == key.N:
            self.index = (self.index + 1) % len(self.samples)
            self.load_current()
        elif symbol == key.P:
            self.index = (self.index - 1) % len(self.samples)
            self.load_current()
        elif symbol == key.O:
            self.open_file_dialog()
        elif symbol == key.T:
            self.use_ac = not self.use_ac
            self.load_current()
        elif symbol == key.X:
            self.flip_x = not self.flip_x
            self.load_current()
        elif symbol == key.H:
            self.spec_strength = 0.15 if self.spec_strength > 0.75 else self.spec_strength + 0.30
            self.update_label()
        elif symbol == key.B:
            self.bump_strength = 0.0 if self.bump_strength > 7.5 else self.bump_strength + 2.5
            self.update_label()
        elif symbol == key.M:
            self.smooth_passes = (self.smooth_passes + 1) % 4
            self.load_current()
        elif symbol == key.F1:
            self.help_visible = not self.help_visible
            self.update_label()
        elif symbol in (key.L,):
            self.light_yaw = (self.light_yaw + 35.0) % 360.0
            self.update_label()
        elif symbol in (key.W, key.UP):
            self.uv_offset[1] += 0.002
            self.update_label()
        elif symbol in (key.S, key.DOWN):
            self.uv_offset[1] -= 0.002
            self.update_label()
        elif symbol in (key.A, key.LEFT):
            self.uv_offset[0] -= 0.002
            self.update_label()
        elif symbol in (key.D, key.RIGHT):
            self.uv_offset[0] += 0.002
            self.update_label()
        elif symbol == key.U:
            self.uv_offset = self.default_uv.copy()
            self.update_label()
        elif symbol == key.E:
            assert self.mesh is not None
            outputs = export_depth_files(self.mesh.sample, Path.cwd(), self.invert_z)
            self.status_label.text = f"Exported depth:\n{outputs[1].name}\n{outputs[0].name}\n{outputs[2].name}\n{outputs[3].name}"
        elif symbol == key.C:
            assert self.mesh is not None
            config_path = Path.cwd() / "ptt_viewer_alignment.txt"
            config_path.write_text(
                "\n".join(
                    [
                        f"sample={self.mesh.sample.name}",
                        f"flipX={self.flip_x}",
                        "depth=inverted",
                        f"uv_x={self.uv_offset[0]:.6f}",
                        f"uv_y={self.uv_offset[1]:.6f}",
                        f"visual_z={self.visual_z:.6f}",
                        f"spec={self.spec_strength:.6f}",
                        f"bump={self.bump_strength:.6f}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.status_label.text = f"Saved alignment:\n{config_path.name}"
        elif symbol == key.R:
            self.yaw, self.tilt, self.zoom = -35.0, 58.0, 1.0
            self.visual_z = 0.65
            self.load_current()
        elif symbol == key.ESCAPE:
            self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenGL textured 3D viewer for Bentron/Pemtron AOI PTT samples.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--grid", type=int, default=360)
    parser.add_argument("--visual-z", type=float, default=0.65)
    parser.add_argument("--uv-x", type=float, default=None, help="Initial texture U offset. Defaults to PTT header registration.")
    parser.add_argument("--uv-y", type=float, default=0.0, help="Initial texture V offset. Positive moves texture up.")
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Open Bentron/Pemtron PTT file",
            filetypes=[("PTT 3D files", "*.ptt"), ("All files", "*.*")],
        )
        root.destroy()
        if not selected:
            return
        selected_path = Path(selected)
        paths = [selected_path.parent]
        initial_name = selected_path.stem
    else:
        initial_name = paths[0].stem if len(paths) == 1 and paths[0].suffix.lower() == ".ptt" else None

    samples = find_samples(paths)
    if not samples:
        if getattr(sys, "frozen", False):
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("PTT Viewer", "No .ptt samples found.")
            root.destroy()
            return
        raise SystemExit("No .ptt samples found.")

    viewer = GLViewer(samples, args.grid, args.visual_z, (args.uv_x, args.uv_y))
    if initial_name:
        for index, sample in enumerate(samples):
            if sample.name == initial_name:
                viewer.index = index
                viewer.load_current()
                break
    pyglet.app.run()


if __name__ == "__main__":
    main()
