from __future__ import annotations

import argparse
import base64
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_ptt(path: Path) -> tuple[int, int, float, float, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 76:
        raise ValueError(f"{path.name}: too small for a PTT header")

    height, width = struct.unpack_from("<II", data, 0)
    x_pitch, y_pitch = struct.unpack_from("<ff", data, 8)
    expected = width * height * 3 * 2
    if len(data) - 76 != expected:
        raise ValueError(f"{path.name}: unexpected PTT payload size")

    planes = np.frombuffer(data, dtype="<u2", offset=76).reshape(3, height, width)
    return width, height, x_pitch, y_pitch, planes


def load_texture(path: Path | None, size: tuple[int, int]) -> np.ndarray:
    if path and path.exists():
        image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)
    return np.dstack([np.full((size[1], size[0]), 170, dtype=np.uint8)] * 3)


def normalize_height(
    height: np.ndarray,
    invalid_threshold: int = 60000,
    z_percentiles: tuple[float, float] = (2.0, 98.0),
    baseline_percentile: float | None = None,
    invert: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    mask = height < invalid_threshold
    valid = height[mask].astype(np.float32)
    if baseline_percentile is None:
        low, high = np.percentile(valid, z_percentiles)
    else:
        low = float(np.percentile(valid, baseline_percentile))
        high = float(np.percentile(valid, z_percentiles[1]))
    if high <= low:
        high = low + 1.0
    z = np.clip((height.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    if invert:
        z = 1.0 - z
    z[~mask] = np.nan
    return z, mask


def smooth_height(z: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return z
    fill = float(np.nanmedian(z[np.isfinite(z)]))
    smoothed = np.nan_to_num(z, nan=fill).astype(np.float32)
    passes = max(1, int(round(radius)))
    for _ in range(passes):
        padded = np.pad(smoothed, ((1, 1), (1, 1)), mode="edge")
        smoothed = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    smoothed[~mask] = np.nan
    return smoothed


def crop_to_content(mask: np.ndarray, margin: int) -> tuple[slice, slice]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(mask.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(mask.shape[1], int(xs.max()) + margin + 1)
    return slice(y0, y1), slice(x0, x1)


def resample(array: np.ndarray, size: tuple[int, int], mode: int) -> np.ndarray:
    if array.dtype == bool:
        image = Image.fromarray(array.astype(np.uint8) * 255)
        return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0
    if np.issubdtype(array.dtype, np.floating):
        fill = np.nanmin(array[np.isfinite(array)])
        safe = np.nan_to_num(array, nan=fill)
        image = Image.fromarray((safe * 65535).astype(np.uint16))
        return np.asarray(image.resize(size, mode), dtype=np.float32) / 65535.0
    image = Image.fromarray(array)
    return np.asarray(image.resize(size, mode))


def shade_texture(texture: np.ndarray, z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = np.nan_to_num(z, nan=np.nanmedian(z[np.isfinite(z)]))
    gy, gx = np.gradient(filled)
    light = np.array([-0.35, -0.45, 0.82], dtype=np.float32)
    normals = np.dstack((-gx * 3.5, -gy * 3.5, np.ones_like(filled)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True).clip(1e-6)
    intensity = np.clip((normals @ light) * 0.65 + 0.45, 0.22, 1.15)
    color = np.clip(texture.astype(np.float32) * intensity[..., None], 0, 255).astype(np.uint8)
    color[~mask] = (20, 24, 28)
    return color


def project(
    x: float,
    y: float,
    z: float,
    scale: float,
    z_scale: float,
    cx: float,
    cy: float,
    yaw_degrees: float,
    pitch_factor: float,
) -> tuple[float, float]:
    angle = math.radians(yaw_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xr = (x - y) * cos_a
    yr = (x + y) * sin_a * pitch_factor - z * z_scale
    return cx + xr * scale, cy + yr * scale


def render_static(
    z: np.ndarray,
    mask: np.ndarray,
    colors: np.ndarray,
    output: Path,
    height_scale: float,
    yaw_degrees: float = 42.0,
    pitch_factor: float = 0.42,
    title: str = "Bentron AOI PTT height surface",
) -> None:
    h, w = z.shape
    canvas_w, canvas_h = 1800, 1200
    scale = min(canvas_w / (w * 1.25), canvas_h / (h * 0.72))
    z_scale = h * height_scale
    cx, cy = canvas_w * 0.50, canvas_h * 0.61
    image = Image.new("RGB", (canvas_w, canvas_h), (14, 18, 22))
    draw = ImageDraw.Draw(image, "RGBA")

    step = 2 if max(w, h) > 180 else 1
    y_range = range(h - 1 - step, 0, -step)
    x_range = range(0, w - 1 - step, step)
    for y in y_range:
        for x in x_range:
            if not (mask[y, x] and mask[y + step, x] and mask[y, x + step] and mask[y + step, x + step]):
                continue
            pts = [
                project(x, y, float(z[y, x]), scale, z_scale, cx, cy, yaw_degrees, pitch_factor),
                project(x + step, y, float(z[y, x + step]), scale, z_scale, cx, cy, yaw_degrees, pitch_factor),
                project(x + step, y + step, float(z[y + step, x + step]), scale, z_scale, cx, cy, yaw_degrees, pitch_factor),
                project(x, y + step, float(z[y + step, x]), scale, z_scale, cx, cy, yaw_degrees, pitch_factor),
            ]
            c = colors[y, x]
            draw.polygon(pts, fill=(int(c[0]), int(c[1]), int(c[2]), 245))

    draw.text((22, 22), title, fill=(230, 236, 240))
    draw.text((22, 48), "Python render: clipped baseline, JPG texture, invalid values hidden", fill=(178, 188, 196))
    image.save(output)


def save_height_debug(z: np.ndarray, mask: np.ndarray, output: Path) -> None:
    img = np.nan_to_num(z, nan=0.0)
    rgb = np.zeros((*img.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(img * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((1.0 - np.abs(img - 0.55) / 0.55) * 220, 0, 220).astype(np.uint8)
    rgb[..., 2] = np.clip((1.0 - img) * 255, 0, 255).astype(np.uint8)
    rgb[~mask] = (0, 0, 0)
    Image.fromarray(rgb).save(output)


def make_webgl_html(z: np.ndarray, mask: np.ndarray, colors: np.ndarray, output: Path, fallback_png: Path, height_scale: float) -> None:
    h, w = z.shape
    vertices: list[float] = []
    vertex_colors: list[float] = []
    indices: list[int] = []
    index_map = np.full((h, w), -1, dtype=np.int32)

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            idx = len(vertices) // 3
            index_map[y, x] = idx
            vertices.extend([(x / (w - 1) - 0.5) * 2.0, (0.5 - y / (h - 1)) * 2.0, float(z[y, x]) * height_scale])
            c = colors[y, x].astype(np.float32) / 255.0
            vertex_colors.extend([float(c[0]), float(c[1]), float(c[2])])

    for y in range(h - 1):
        for x in range(w - 1):
            a, b, c, d = index_map[y, x], index_map[y, x + 1], index_map[y + 1, x], index_map[y + 1, x + 1]
            if min(a, b, c, d) >= 0:
                indices.extend([int(a), int(c), int(b), int(b), int(c), int(d)])

    payload = {
        "vertices": vertices,
        "colors": vertex_colors,
        "indices": indices,
        "indexType": "uint16" if max(indices, default=0) < 65536 else "uint32",
    }
    encoded = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
    html = f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>Bentron AOI PTT 3D Preview</title>
<style>
html, body {{ margin: 0; height: 100%; background: #11161b; color: #d9e2ea; font-family: Arial, sans-serif; overflow: hidden; }}
body {{ background: #10151a; }}
#fallback {{ position: fixed; inset: 0; width: 100vw; height: 100vh; object-fit: contain; background: #10151a; }}
#hint, #status {{ position: fixed; left: 16px; padding: 8px 10px; background: rgba(0,0,0,.50); border-radius: 6px; font-size: 13px; z-index: 3; }}
#hint {{ top: 14px; }}
#status {{ bottom: 14px; color: #ffcc7a; }}
canvas {{ position: fixed; inset: 0; width: 100vw; height: 100vh; display: block; z-index: 2; }}
</style>
<img id="fallback" src="{fallback_png.name}" alt="static 3D preview">
<canvas id="gl"></canvas>
<div id="hint">拖拽旋转，滚轮缩放。Height: PTT plane0, Texture: JPG</div>
<div id="status">正在加载交互 3D；如果 WebGL 不可用，会保留这张静态 3D 预览。</div>
<script>
const statusEl = document.getElementById("status");
const fallbackEl = document.getElementById("fallback");
try {{
const data = JSON.parse(atob("{encoded}"));
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl", {{antialias: true}});
if (!gl) throw new Error("当前浏览器没有启用 WebGL");
const vs = `
attribute vec3 position;
attribute vec3 color;
uniform float rx;
uniform float rz;
uniform float zoom;
uniform float aspect;
varying vec3 vColor;
void main() {{
  vColor = color;
  float cz = cos(rz), sz = sin(rz);
  float cx = cos(rx), sx = sin(rx);
  vec3 p = vec3(position.x * cz - position.y * sz, position.x * sz + position.y * cz, position.z);
  p = vec3(p.x, p.y * cx - p.z * sx, p.y * sx + p.z * cx);
  float s = zoom;
  gl_Position = vec4(p.x * s / aspect, p.y * s, 0.2 + p.z * 0.08, 1.0);
}}
`;
const fs = `
precision mediump float;
varying vec3 vColor;
void main() {{
  gl_FragColor = vec4(vColor, 1.0);
}}
`;
function shader(type, source) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}}
const program = gl.createProgram();
gl.attachShader(program, shader(gl.VERTEX_SHADER, vs));
gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(program);
gl.useProgram(program);
function buffer(attr, values) {{
  const b = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(values), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, attr);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
}}
buffer("position", data.vertices);
buffer("color", data.colors);
const ib = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
let indexArray, indexEnum;
if (data.indexType === "uint16") {{
  indexArray = new Uint16Array(data.indices);
  indexEnum = gl.UNSIGNED_SHORT;
}} else {{
  const ext = gl.getExtension("OES_element_index_uint");
  if (!ext) throw new Error("当前 WebGL 环境不支持 32-bit 网格索引，请用更小的 --grid 重新生成");
  indexArray = new Uint32Array(data.indices);
  indexEnum = gl.UNSIGNED_INT;
}}
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexArray, gl.STATIC_DRAW);
let rx = -0.95, rz = -0.72, zoom = 0.86, dragging = false, lx = 0, ly = 0;
canvas.addEventListener("pointerdown", e => {{ dragging = true; lx = e.clientX; ly = e.clientY; }});
canvas.addEventListener("pointerup", () => dragging = false);
canvas.addEventListener("pointermove", e => {{
  if (!dragging) return;
  rz += (e.clientX - lx) * 0.008;
  rx += (e.clientY - ly) * 0.008;
  lx = e.clientX; ly = e.clientY;
}});
canvas.addEventListener("wheel", e => {{ e.preventDefault(); zoom = Math.max(0.25, Math.min(2.5, zoom - e.deltaY * 0.001)); }}, {{passive:false}});
function render() {{
  canvas.width = Math.floor(innerWidth * devicePixelRatio);
  canvas.height = Math.floor(innerHeight * devicePixelRatio);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.067, 0.086, 0.106, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.uniform1f(gl.getUniformLocation(program, "rx"), rx);
  gl.uniform1f(gl.getUniformLocation(program, "rz"), rz);
  gl.uniform1f(gl.getUniformLocation(program, "zoom"), zoom);
  gl.uniform1f(gl.getUniformLocation(program, "aspect"), canvas.width / canvas.height);
  gl.drawElements(gl.TRIANGLES, data.indices.length, indexEnum, 0);
  fallbackEl.style.display = "none";
  requestAnimationFrame(render);
}}
statusEl.textContent = `vertices: ${{data.vertices.length / 3}}, triangles: ${{data.indices.length / 3}}, index: ${{data.indexType}}`;
render();
}} catch (err) {{
  statusEl.textContent = "3D 渲染失败: " + err.message;
  document.getElementById("gl").style.display = "none";
  console.error(err);
}}
</script>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 3D previews from Bentron AOI PTT files.")
    parser.add_argument("ptt", type=Path)
    parser.add_argument("--texture", type=Path, default=None)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("renders_3d"))
    parser.add_argument("--grid", type=int, default=140, help="Maximum mesh width/height for 3D preview.")
    parser.add_argument("--height-scale", type=float, default=0.22, help="Relative Z scale. Increase for stronger 3D relief.")
    parser.add_argument("--plane", type=int, default=0, choices=[0, 1, 2], help="PTT plane to use as height.")
    parser.add_argument("--z-percentiles", type=float, nargs=2, default=(2.0, 99.5), metavar=("LOW", "HIGH"))
    parser.add_argument("--baseline-percentile", type=float, default=55.0, help="Raw height percentile treated as board/base level.")
    parser.add_argument("--smooth", type=float, default=1.2, help="Gaussian smoothing radius after normalization.")
    parser.add_argument("--invert", action="store_true", help="Invert normalized height.")
    args = parser.parse_args()

    width, height, x_pitch, y_pitch, planes = read_ptt(args.ptt)
    z, mask = normalize_height(
        planes[args.plane],
        z_percentiles=(args.z_percentiles[0], args.z_percentiles[1]),
        baseline_percentile=args.baseline_percentile,
        invert=args.invert,
    )
    y_slice, x_slice = crop_to_content(mask, margin=16)
    z = z[y_slice, x_slice]
    mask = mask[y_slice, x_slice]
    z = smooth_height(z, mask, args.smooth)
    texture = load_texture(args.texture, (width, height))[y_slice, x_slice]

    crop_h, crop_w = z.shape
    scale = min(1.0, args.grid / max(crop_w, crop_h))
    mesh_size = (max(4, int(crop_w * scale)), max(4, int(crop_h * scale)))
    z_small = resample(z, mesh_size, Image.Resampling.BILINEAR)
    mask_small = resample(mask, mesh_size, Image.Resampling.NEAREST)
    texture_small = resample(texture, mesh_size, Image.Resampling.BILINEAR)
    colors = shade_texture(texture_small, z_small, mask_small)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / f"{args.ptt.stem}_3d_preview.png"
    html = args.output_dir / f"{args.ptt.stem}_3d_interactive.html"
    debug_png = args.output_dir / f"{args.ptt.stem}_height_debug_plane{args.plane}.png"
    save_height_debug(z_small, mask_small, debug_png)
    render_static(
        z_small,
        mask_small,
        colors,
        png,
        args.height_scale,
        yaw_degrees=42.0,
        title=f"Bentron AOI PTT plane{args.plane} Python 3D",
    )
    for yaw in (20.0, 70.0, 120.0):
        angle_png = args.output_dir / f"{args.ptt.stem}_3d_yaw{int(yaw)}.png"
        render_static(
            z_small,
            mask_small,
            colors,
            angle_png,
            args.height_scale,
            yaw_degrees=yaw,
            title=f"Bentron AOI PTT plane{args.plane} yaw {int(yaw)}",
        )
    make_webgl_html(z_small, mask_small, colors, html, png, args.height_scale)

    print(f"{args.ptt.name}: source={width}x{height}, crop={crop_w}x{crop_h}, mesh={mesh_size[0]}x{mesh_size[1]}")
    print(f"pitch: x={x_pitch:.8f}, y={y_pitch:.8f}")
    print(
        f"height: plane={args.plane}, z_percentiles={args.z_percentiles[0]:.1f}/{args.z_percentiles[1]:.1f}, "
        f"baseline_percentile={args.baseline_percentile:.1f}, smooth={args.smooth:.1f}, invert={args.invert}"
    )
    print(f"static preview: {png.resolve()}")
    print(f"height debug: {debug_png.resolve()}")
    print(f"interactive html: {html.resolve()}")


if __name__ == "__main__":
    main()
