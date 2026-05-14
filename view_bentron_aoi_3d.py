from __future__ import annotations

import argparse
import math
import struct
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk


@dataclass
class Sample:
    name: str
    ptt: Path
    pot: Path | None
    jpg: Path | None
    ac_jpg: Path | None


@dataclass
class Mesh:
    name: str
    width: int
    height: int
    pitch_x: float
    pitch_y: float
    baseline_raw: float
    z_world: np.ndarray
    mask: np.ndarray
    colors: np.ndarray
    auto_z_scale: float


def find_samples(paths: list[Path]) -> list[Sample]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.ptt")))
        elif path.suffix.lower() == ".ptt":
            files.append(path)

    samples: list[Sample] = []
    seen: set[Path] = set()
    for ptt in files:
        ptt = ptt.resolve()
        if ptt in seen:
            continue
        seen.add(ptt)
        stem = ptt.with_suffix("")
        samples.append(
            Sample(
                name=ptt.stem,
                ptt=ptt,
                pot=stem.with_suffix(".pot") if stem.with_suffix(".pot").exists() else None,
                jpg=stem.with_suffix(".jpg") if stem.with_suffix(".jpg").exists() else None,
                ac_jpg=Path(str(stem) + "_AC.jpg") if Path(str(stem) + "_AC.jpg").exists() else None,
            )
        )
    return sorted(samples, key=lambda item: item.name)


def read_ptt(path: Path) -> tuple[int, int, float, float, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 76:
        raise ValueError(f"{path.name}: too small")
    height, width = struct.unpack_from("<II", data, 0)
    pitch_x, pitch_y = struct.unpack_from("<ff", data, 8)
    expected = width * height * 3 * 2
    if len(data) - 76 != expected:
        raise ValueError(f"{path.name}: expected {expected} payload bytes, got {len(data) - 76}")
    planes = np.frombuffer(data, dtype="<u2", offset=76).reshape(3, height, width)
    return width, height, pitch_x, pitch_y, planes


def read_pot_planes(path: Path, width: int, height: int) -> np.ndarray:
    data = path.read_bytes()
    expected = width * height * 5
    if len(data) - 20 != expected:
        raise ValueError(f"{path.name}: unexpected POT payload size")
    return np.frombuffer(data, dtype=np.uint8, offset=20).reshape(5, height, width)


def resize_array(array: np.ndarray, size: tuple[int, int], resampling: int) -> np.ndarray:
    if array.dtype == bool:
        image = Image.fromarray(array.astype(np.uint8) * 255)
        return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0
    if np.issubdtype(array.dtype, np.floating):
        fill = float(np.nanmedian(array[np.isfinite(array)]))
        safe = np.nan_to_num(array, nan=fill).astype(np.float32)
        image = Image.fromarray(safe, mode="F")
        return np.asarray(image.resize(size, resampling), dtype=np.float32)
    return np.asarray(Image.fromarray(array).resize(size, resampling))


def shade(texture: np.ndarray, z_world: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = np.nan_to_num(z_world, nan=float(np.nanmedian(z_world[np.isfinite(z_world)])))
    # Normalize gradients so lighting is stable across tall and short parts.
    denom = max(1.0, float(np.percentile(filled[mask], 98)))
    gy, gx = np.gradient(filled / denom)
    normals = np.dstack((-gx * 4.0, -gy * 4.0, np.ones_like(filled)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True).clip(1e-6)
    light = np.array([-0.35, -0.45, 0.82], dtype=np.float32)
    intensity = np.clip((normals @ light) * 0.36 + 0.72, 0.42, 1.12)
    colors = np.clip(texture.astype(np.float32) * intensity[..., None], 0, 255).astype(np.uint8)
    colors[~mask] = (18, 22, 26)
    return colors


def normalize_u8(channel: np.ndarray) -> np.ndarray:
    data = channel.astype(np.float32)
    low, high = np.percentile(data, [1, 99])
    if high <= low:
        high = low + 1
    return np.clip((data - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def build_texture(sample: Sample, width: int, height: int, texture_mode: str) -> np.ndarray:
    base_path = sample.jpg or sample.ac_jpg
    if base_path:
        base_image = Image.open(base_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        base_image = base_image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=3))
        base = np.asarray(base_image, dtype=np.uint8)
    else:
        base = np.full((height, width, 3), 170, dtype=np.uint8)

    if texture_mode == "ac" and sample.ac_jpg:
        return np.asarray(
            Image.open(sample.ac_jpg).convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

    if sample.pot and texture_mode in {"blend", "pot0", "pot4"}:
        pot = read_pot_planes(sample.pot, width, height)
        plane = normalize_u8(pot[4] if texture_mode == "pot4" else pot[0])
        if texture_mode.startswith("pot"):
            return np.dstack([plane, plane, plane])

        # Preserve JPG color, but replace some luminance with the POT grayscale detail.
        base_f = base.astype(np.float32)
        gray = (base_f[..., 0] * 0.299 + base_f[..., 1] * 0.587 + base_f[..., 2] * 0.114).clip(1, 255)
        detail = plane.astype(np.float32).clip(1, 255)
        factor = np.clip((detail / gray) ** 0.55, 0.55, 1.45)
        return np.clip(base_f * factor[..., None], 0, 255).astype(np.uint8)

    return base


def load_mesh(sample: Sample, grid: int, baseline_pct: float, high_clip_pct: float, texture_mode: str) -> Mesh:
    width, height, pitch_x, pitch_y, planes = read_ptt(sample.ptt)
    raw = planes[0].astype(np.float32)
    mask = raw < 60000
    valid = raw[mask]
    baseline = float(np.percentile(valid, baseline_pct))
    high = float(np.percentile(valid, high_clip_pct))
    raw_delta = np.clip(raw - baseline, 0.0, max(1.0, high - baseline))

    pitch = max(1e-6, (pitch_x + pitch_y) * 0.5)
    # Convert height into the same unit as XY pixels: one XY pixel is about pitch microns.
    z_world = raw_delta / pitch
    z_world[~mask] = np.nan

    scale = min(1.0, grid / max(width, height))
    mesh_size = (max(8, int(width * scale)), max(8, int(height * scale)))
    z_small = resize_array(z_world, mesh_size, Image.Resampling.BILINEAR) * scale
    mask_small = resize_array(mask, mesh_size, Image.Resampling.NEAREST)

    texture = build_texture(sample, width, height, texture_mode)
    texture_small = resize_array(texture, mesh_size, Image.Resampling.BILINEAR)
    colors = shade(texture_small, z_small, mask_small)

    # Keep one fixed visual Z scale across samples. Per-sample auto-fit makes
    # short parts such as sample 206 look falsely tall.
    auto_z_scale = 0.65
    return Mesh(sample.name, width, height, pitch_x, pitch_y, baseline, z_small, mask_small, colors, auto_z_scale)


class Viewer:
    def __init__(self, root: tk.Tk, samples: list[Sample], grid: int, baseline_pct: float, high_clip_pct: float) -> None:
        self.root = root
        self.samples = samples
        self.grid = grid
        self.baseline_pct = baseline_pct
        self.high_clip_pct = high_clip_pct
        self.texture_modes = ["jpg", "blend", "pot0", "pot4", "ac"]
        self.texture_mode = "jpg"
        self.index = 0
        self.mesh: Mesh | None = None
        self.yaw = 42.0
        self.tilt = 58.0
        self.zoom = 1.0
        self.z_scale = 1.0
        self.last: tuple[int, int] | None = None
        self.photo: ImageTk.PhotoImage | None = None

        root.title("Bentron AOI Python 3D Viewer")
        root.geometry("1320x860")
        root.bind("<KeyPress>", self.on_key)

        self.sidebar = tk.Frame(root, width=220)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox = tk.Listbox(self.sidebar, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        for sample in samples:
            self.listbox.insert(tk.END, sample.name)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        button_row = tk.Frame(self.sidebar)
        button_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Button(button_row, text="Prev", command=lambda: self.change_sample(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(button_row, text="Next", command=lambda: self.change_sample(1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(self.sidebar, text="Texture mode", command=self.toggle_texture).pack(fill=tk.X, padx=6, pady=(0, 6))
        self.info = tk.Label(self.sidebar, justify=tk.LEFT, anchor="nw", text="", wraplength=205)
        self.info.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.canvas = tk.Canvas(root, bg="#0f1419", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.render())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        self.listbox.selection_set(0)
        self.load_current(reset_view=True)

    def load_current(self, reset_view: bool = False) -> None:
        self.mesh = load_mesh(self.samples[self.index], self.grid, self.baseline_pct, self.high_clip_pct, self.texture_mode)
        if reset_view:
            self.yaw, self.tilt, self.zoom = 42.0, 58.0, 1.0
        self.z_scale = self.mesh.auto_z_scale
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.index)
        self.listbox.see(self.index)
        self.update_info()
        self.render()

    def update_info(self) -> None:
        assert self.mesh is not None
        z_valid = self.mesh.z_world[self.mesh.mask]
        self.info.config(
            text=(
                f"{self.mesh.name}\n"
                f"source: {self.mesh.width} x {self.mesh.height}\n"
                f"mesh: {self.mesh.z_world.shape[1]} x {self.mesh.z_world.shape[0]}\n"
                f"pitch: {self.mesh.pitch_x:.5f}, {self.mesh.pitch_y:.5f}\n"
                f"baseline raw: {self.mesh.baseline_raw:.1f}\n"
                f"height p95/p99: {np.percentile(z_valid, 95):.1f}/{np.percentile(z_valid, 99):.1f} px-unit\n"
                f"visual_z: {self.z_scale:.3f}\n"
                f"texture: {self.texture_mode}"
            )
        )

    def on_select(self, _event: tk.Event) -> None:
        selected = self.listbox.curselection()
        if not selected:
            return
        new_index = int(selected[0])
        if new_index != self.index:
            self.index = new_index
            self.load_current(reset_view=True)

    def change_sample(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.samples)
        self.load_current(reset_view=True)

    def toggle_texture(self) -> None:
        current = self.texture_modes.index(self.texture_mode)
        self.texture_mode = self.texture_modes[(current + 1) % len(self.texture_modes)]
        self.load_current(reset_view=False)

    def on_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        self.last = (event.x, event.y)

    def on_release(self, _event: tk.Event) -> None:
        self.last = None

    def on_drag(self, event: tk.Event) -> None:
        if self.last is None:
            self.last = (event.x, event.y)
            return
        lx, ly = self.last
        self.yaw += (event.x - lx) * 0.35
        self.tilt = min(88.0, max(18.0, self.tilt + (event.y - ly) * 0.20))
        self.last = (event.x, event.y)
        self.render()

    def on_wheel(self, event: tk.Event) -> None:
        self.zoom = min(4.0, max(0.25, self.zoom * (1.1 if event.delta > 0 else 0.9)))
        self.render()

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in ("plus", "equal"):
            self.z_scale *= 1.15
        elif key in ("minus", "underscore"):
            self.z_scale /= 1.15
        elif key == "r":
            self.yaw, self.tilt, self.zoom = 42.0, 58.0, 1.0
            if self.mesh:
                self.z_scale = self.mesh.auto_z_scale
        elif key == "1":
            self.z_scale = 0.35
        elif key == "2":
            self.z_scale = 0.65
        elif key == "3":
            self.z_scale = 0.95
        elif key in ("n", "right"):
            self.change_sample(1)
            return
        elif key in ("p", "left"):
            self.change_sample(-1)
            return
        elif key == "t":
            self.toggle_texture()
            return
        elif key == "escape":
            self.root.destroy()
            return
        self.update_info()
        self.render()

    def transform(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        assert self.mesh is not None
        h, w = self.mesh.z_world.shape
        centered_x = x - w * 0.5
        centered_y = y - h * 0.5
        centered_z = z * self.z_scale

        yaw = math.radians(self.yaw)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        x1 = centered_x * cos_yaw - centered_y * sin_yaw
        y1 = centered_x * sin_yaw + centered_y * cos_yaw

        tilt = math.radians(self.tilt)
        cos_tilt, sin_tilt = math.cos(tilt), math.sin(tilt)
        screen_y = y1 * cos_tilt - centered_z * sin_tilt
        depth = y1 * sin_tilt + centered_z * cos_tilt
        return x1, screen_y, depth

    def render(self) -> None:
        if self.mesh is None:
            return
        z = self.mesh.z_world
        mask = self.mesh.mask
        colors = self.mesh.colors
        h, w = z.shape
        canvas_w = max(640, self.canvas.winfo_width())
        canvas_h = max(480, self.canvas.winfo_height())
        image = Image.new("RGB", (canvas_w, canvas_h), (14, 18, 22))
        draw = ImageDraw.Draw(image, "RGBA")

        scale = min(canvas_w / (w * 1.25), canvas_h / (h * 0.76)) * self.zoom
        cx, cy = canvas_w * 0.50, canvas_h * 0.62
        step = 2 if max(w, h) > 190 else 1
        polygons: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []

        for y in range(h - 1 - step, 0, -step):
            for x in range(0, w - 1 - step, step):
                if not (mask[y, x] and mask[y + step, x] and mask[y, x + step] and mask[y + step, x + step]):
                    continue
                transformed = [
                    self.transform(x, y, float(z[y, x])),
                    self.transform(x + step, y, float(z[y, x + step])),
                    self.transform(x + step, y + step, float(z[y + step, x + step])),
                    self.transform(x, y + step, float(z[y + step, x])),
                ]
                pts = [(cx + px * scale, cy + py * scale) for px, py, _ in transformed]
                depth = sum(point[2] for point in transformed) / 4.0
                c = colors[y, x]
                polygons.append((depth, pts, (int(c[0]), int(c[1]), int(c[2]))))

        for _depth, pts, color in sorted(polygons, key=lambda item: item[0]):
            draw.polygon(pts, fill=(*color, 245))

        draw.text((14, 12), "drag rotate | wheel zoom | +/- visual Z | 1/2/3 height presets | n/p sample | t texture | r reset", fill=(226, 234, 240))
        draw.text((14, 34), f"{self.mesh.name}  yaw={self.yaw:.1f} tilt={self.tilt:.1f} zoom={self.zoom:.2f} visual_z={self.z_scale:.3f}", fill=(186, 198, 208))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Python software renderer for Bentron/Pemtron AOI PTT files.")
    parser.add_argument("paths", nargs="*", type=Path, help="PTT files or directories containing PTT samples.")
    parser.add_argument("--grid", type=int, default=240)
    parser.add_argument("--baseline-percentile", type=float, default=20.0)
    parser.add_argument("--high-clip-percentile", type=float, default=99.5)
    args = parser.parse_args()

    paths = args.paths or [Path("samples"), Path(".")]
    samples = find_samples(paths)
    if not samples:
        raise SystemExit("No .ptt samples found.")

    root = tk.Tk()
    Viewer(root, samples, args.grid, args.baseline_percentile, args.high_clip_percentile)
    root.mainloop()


if __name__ == "__main__":
    main()
