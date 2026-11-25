"""胶片颗粒时域合成管线。

针对六个缺陷的修复对应关系：

1. 时域相关 —— :func:`temporal_correlation`
   颗粒不再逐帧独立白噪，而是在低分辨率晶格上跑 AR(1) 过程
   ``g_t = alpha * g_{t-1} + sqrt(1-alpha^2) * epsilon_t``，
   再双线性上采样到像素分辨率。静止画面颗粒连续游走，不再闪烁，
   压缩码率随之回落。

2. 分区强度 —— :func:`strength_by_region`
   强度按分区局部亮度计算：暗区保持高增益（不被压没），
   高光区压低增益（不爆成白点），替代整帧平均亮度的单值强度。

3. 镜头种子 —— :func:`seed_per_shot`
   每个镜头由镜头 ID 经 splitmix64 派生独立 64 位种子，
   帧/晶格单元的随机流全部由 (镜头种子, 帧号) 哈希派生，
   不使用任何全局 RNG。多机位并行、同镜头重复渲染结果逐位一致。

4. 映射前叠加 —— :func:`grain_before_tonemap`
   颗粒在线性光空间、色调映射之前叠加，HDR 高光（>1）带着
   颗粒一起进入映射曲线，而不是先裁到 1 再加颗粒（方块来源）。

5. 物理尺寸 —— :func:`size_by_physical`
   颗粒直径以毫米给出，按传感器宽度换算成像素晶格间距，
   分辨率变化时颗粒在画面中的物理尺寸保持恒定。

6. 软限幅 —— :func:`soft_limit`
   对称 filmic 软膝 ``sign(x) * white * |x| / (|x| + white)``，
   C1 连续、永不到达硬边界，高光不再出现硬裁切方块。

管线顺序（见 :func:`render`）：

    镜头种子 -> 时域晶格颗粒(物理尺寸) -> 分区强度
             -> 线性空间叠加(映射前) -> 软限幅 -> 色调映射 -> 显示图

所有公开函数在第一个位置参数为整数时进入内置演示路径，
因此可以零参数依赖地单独调用（例如 ``soft_limit(1)``）。
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

__all__ = [
    "temporal_correlation",
    "strength_by_region",
    "seed_per_shot",
    "grain_before_tonemap",
    "size_by_physical",
    "soft_limit",
    "generate",
    "sample",
    "composite",
    "render",
    "export",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

_U64_MASK = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3

# 管线默认参数
DEFAULT_SHAPE = (32, 32)
DEFAULT_ALPHA = 0.85              # 相邻帧颗粒相关系数
DEFAULT_BASE_STRENGTH = 0.10      # 线性光颗粒基准幅度
DEFAULT_ZONES = 8
DEFAULT_GRAIN_DIAMETER_MM = 0.02     # 物理颗粒直径；1080p≈1px、4K≈2px
DEFAULT_SENSOR_WIDTH_MM = 36.0    # 全画幅
DEFAULT_PIVOT = 0.18              # 中灰拐点（线性光）


# ---------------------------------------------------------------------------
# 基础工具：确定性哈希与每镜头 RNG
# ---------------------------------------------------------------------------

def _splitmix64(x: int) -> int:
    """splitmix64 单步混合，返回 64 位非负整数。"""
    x = (x + 0x9E3779B97F4A7C15) & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return x ^ (x >> 31)


def _fnv1a64(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & _U64_MASK
    return h


def _hash_part(part) -> int:
    if isinstance(part, str):
        return _fnv1a64(part.encode("utf-8"))
    if isinstance(part, (bytes, bytearray)):
        return _fnv1a64(bytes(part))
    return int(part) & _U64_MASK


def mix_seed(*parts) -> int:
    """把若干组成部分混合成一个 64 位种子（顺序敏感）。"""
    h = 0
    for part in parts:
        h = _splitmix64(h ^ _hash_part(part))
    return h


def seed_per_shot(shot_id) -> int:
    """由镜头 ID 派生稳定的 64 位随机种子。

    同一镜头 ID 永远得到同一种子；不同镜头（含字符串名）相互独立。
    不触碰任何全局随机状态，因此多机位并行渲染不会互相消耗序列。
    """
    if isinstance(shot_id, (int, np.integer)):
        return _splitmix64(int(shot_id) & _U64_MASK)
    return _splitmix64(_hash_part(shot_id))


def _frame_rng(shot_seed: int, frame_index: int) -> np.random.Generator:
    """帧级 RNG：仅由 (镜头种子, 帧号) 决定，与调用顺序无关。"""
    return np.random.default_rng(mix_seed(shot_seed, int(frame_index)))


# ---------------------------------------------------------------------------
# 修复 5：物理尺寸
# ---------------------------------------------------------------------------

def size_by_physical(width, height=None,
                     grain_diameter_mm=DEFAULT_GRAIN_DIAMETER_MM,
                     sensor_width_mm=DEFAULT_SENSOR_WIDTH_MM):
    """把物理颗粒直径换算成像素晶格间距。

    ``px_per_mm = width / sensor_width_mm``，故
    ``lattice_px = grain_diameter_mm * width / sensor_width_mm``。
    分辨率改变时晶格间距按比例缩放，颗粒在画面中的物理大小恒定。
    """
    if isinstance(width, (int, np.integer)) and height is None:
        # 演示调用：size_by_physical(1)
        height = max(1, int(round(width * 9 / 16)))
    height = int(height) if height is not None else int(round(width * 9 / 16))
    width = int(width)
    px_per_mm = width / sensor_width_mm
    diameter_px = grain_diameter_mm * px_per_mm
    lattice_px = max(1, int(round(diameter_px)))
    return {
        "width": width,
        "height": height,
        "grain_diameter_mm": float(grain_diameter_mm),
        "sensor_width_mm": float(sensor_width_mm),
        "px_per_mm": float(px_per_mm),
        "radius_px": float(diameter_px * 0.5),
        "lattice_px": lattice_px,
    }


# ---------------------------------------------------------------------------
# 修复 1：时域相关（AR(1) 晶格 + 双线性上采样）
# ---------------------------------------------------------------------------

def _lattice_shape(shape, lattice_px):
    h, w = shape
    return max(1, int(np.ceil(h / lattice_px))), max(1, int(np.ceil(w / lattice_px)))


def _ar1_lattice(shot_seed, frame_index, lattice_shape, alpha):
    """无状态回放 AR(1) 晶格到第 ``frame_index`` 帧。

    每帧新息由帧级 RNG 独立生成，回放任意帧只依赖 (镜头种子, 帧号)，
    不需要保留上一帧，因此并行 worker 与重渲染结果一致。
    """
    alpha = float(alpha)
    innovation_sigma = float(np.sqrt(max(0.0, 1.0 - alpha * alpha)))
    g = _frame_rng(shot_seed, 0).standard_normal(lattice_shape)
    for t in range(1, int(frame_index) + 1):
        eps = _frame_rng(shot_seed, t).standard_normal(lattice_shape)
        g = alpha * g + innovation_sigma * eps
    return g


def _upsample_bilinear(field, out_shape):
    """以像素中心对齐的双线性插值把晶格场放大到 ``out_shape``。"""
    h, w = field.shape
    out_h, out_w = out_shape
    if (h, w) == (out_h, out_w):
        return field.copy()
    fy = (np.arange(out_h, dtype=np.float64) + 0.5) * (h / out_h) - 0.5
    fx = (np.arange(out_w, dtype=np.float64) + 0.5) * (w / out_w) - 0.5
    y0 = np.floor(fy).astype(np.int64)
    x0 = np.floor(fx).astype(np.int64)
    wy, wx = fy - y0, fx - x0
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    top = field[np.ix_(y0c, x0c)] * (1 - wx) + field[np.ix_(y0c, x1c)] * wx
    bot = field[np.ix_(y1c, x0c)] * (1 - wx) + field[np.ix_(y1c, x1c)] * wx
    return top * (1 - wy)[:, None] + bot * wy[:, None]


def temporal_correlation(frame_index, shot_seed=0, shape=DEFAULT_SHAPE,
                         alpha=DEFAULT_ALPHA, lattice_px=4):
    """返回第 ``frame_index`` 帧的时域相关颗粒场（像素分辨率，零均值单位方差量级）。

    相邻帧的 Pearson 相关系数稳态等于 ``alpha``；颗粒空间团块大小
    由 ``lattice_px``（物理尺寸换算结果）决定。
    """
    if isinstance(frame_index, (int, np.integer)) and isinstance(shot_seed, (int, np.integer)):
        pass  # 常规/演示调用均走此路径
    shape = tuple(shape)
    lattice_shape = _lattice_shape(shape, max(1, int(lattice_px)))
    lattice = _ar1_lattice(int(shot_seed), int(frame_index), lattice_shape, alpha)
    grain = _upsample_bilinear(lattice, shape)
    # 上采样会衰减方差，按标准差归一化回单位强度，保证强度控制可预期
    std = grain.std()
    if std > 1e-12:
        grain = grain / std
    return grain


# ---------------------------------------------------------------------------
# 修复 2：分区强度
# ---------------------------------------------------------------------------

def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def strength_by_region(luminance, base_strength=DEFAULT_BASE_STRENGTH,
                       shadow_gain=1.6, highlight_gain=0.45,
                       zones=DEFAULT_ZONES, pivot=DEFAULT_PIVOT):
    """按分区平均亮度计算逐像素颗粒强度。

    返回 ``(strength_field, zone_factors)``：暗区增益接近
    ``shadow_gain``，随分区亮度经 smoothstep 过渡到 ``highlight_gain``。
    暗场颗粒保留、亮场颗粒收敛，不再用全帧平均亮度一刀切。
    """
    if isinstance(luminance, (int, np.integer)):
        luminance = _demo_image(32, 32)
    lum = np.asarray(luminance, dtype=np.float64)
    h, w = lum.shape
    n = max(1, int(zones))
    zone_luma = np.zeros((n, n), dtype=np.float64)
    band_h = max(1, h // n)
    band_w = max(1, w // n)
    for j in range(n):
        ys = slice(j * band_h, h if j == n - 1 else (j + 1) * band_h)
        for i in range(n):
            xs = slice(i * band_w, w if i == n - 1 else (i + 1) * band_w)
            zone_luma[j, i] = lum[ys, xs].mean()
    t = _smoothstep(zone_luma / float(pivot))
    zone_factors = shadow_gain + (highlight_gain - shadow_gain) * t
    factor_field = _upsample_bilinear(zone_factors, (h, w))
    return float(base_strength) * factor_field, zone_factors


# ---------------------------------------------------------------------------
# 修复 4：映射前叠加
# ---------------------------------------------------------------------------

def grain_before_tonemap(linear_image, grain=None, strength=None):
    """在色调映射之前于线性光空间叠加颗粒。

    结果允许超过 1（HDR 高光连同颗粒一起交给后续映射/软限），
    从根上消除“先裁到 1 再叠颗粒”造成的高光方块。
    """
    if isinstance(linear_image, (int, np.integer)):
        linear_image = _demo_image(32, 32)
        grain = temporal_correlation(0, seed_per_shot(1), linear_image.shape)
        strength, _ = strength_by_region(_luma(linear_image))
    base = np.asarray(linear_image, dtype=np.float64)
    if grain is None:
        grain = np.zeros_like(base)
    grain = np.asarray(grain, dtype=np.float64)
    if strength is None:
        strength = 1.0
    return base + grain * np.asarray(strength, dtype=np.float64)


# ---------------------------------------------------------------------------
# 修复 6：软限幅
# ---------------------------------------------------------------------------

def soft_limit(x, white=1.0):
    """对称 filmic 软膝：``sign(x) * white * |x| / (|x| + white)``。

    把整条实轴单调压进 (-white, white)，C1 连续且永不到达硬边界，
    正负颗粒均被平滑收纳，不会在 0/1 处产生硬裁切点。
    """
    if isinstance(x, (int, np.integer, float)):
        x = np.float64(x)
    x = np.asarray(x, dtype=np.float64)
    white = float(white)
    return np.sign(x) * white * np.abs(x) / (np.abs(x) + white)


def _tonemap_aces(linear):
    """ACES 近似色调映射（输入线性光，输出显示参考空间）。"""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.asarray(linear, dtype=np.float64)
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 演示数据与小工具
# ---------------------------------------------------------------------------

def _demo_image(h=32, w=32):
    """构造含暗区、中灰与 HDR 高光（>1）的演示线性光画面。"""
    ramp = np.linspace(0.02, 2.0, w, dtype=np.float64)
    img = np.tile(ramp, (h, 1))
    img[: h // 2, :] *= 0.25
    return img


def _luma(linear_image):
    img = np.asarray(linear_image, dtype=np.float64)
    if img.ndim == 3:
        return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    return img


# ---------------------------------------------------------------------------
# 颗粒生成与采样
# ---------------------------------------------------------------------------

@dataclass
class GrainField:
    shot_id: object
    shot_seed: int
    frame: int
    shape: tuple
    lattice_px: int
    alpha: float
    grain: np.ndarray


def generate(shot_id, frame_index=0, shape=DEFAULT_SHAPE, alpha=DEFAULT_ALPHA,
             grain_diameter_mm=DEFAULT_GRAIN_DIAMETER_MM,
             sensor_width_mm=DEFAULT_SENSOR_WIDTH_MM):
    """生成单镜头单帧的时域相关颗粒场（物理尺寸、镜头种子隔离）。"""
    if isinstance(shot_id, (int, np.integer)) and not isinstance(shot_id, bool):
        pass
    shape = tuple(shape)
    shot_seed = seed_per_shot(shot_id)
    size = size_by_physical(shape[1], shape[0],
                            grain_diameter_mm=grain_diameter_mm,
                            sensor_width_mm=sensor_width_mm)
    grain = temporal_correlation(frame_index, shot_seed, shape,
                                 alpha=alpha, lattice_px=size["lattice_px"])
    return GrainField(shot_id=shot_id, shot_seed=shot_seed,
                      frame=int(frame_index), shape=shape,
                      lattice_px=size["lattice_px"], alpha=float(alpha),
                      grain=grain)


def sample(linear_image, frame_index=0, shot_id=0,
           alpha=DEFAULT_ALPHA, base_strength=DEFAULT_BASE_STRENGTH,
           zones=DEFAULT_ZONES,
           grain_diameter_mm=DEFAULT_GRAIN_DIAMETER_MM,
           sensor_width_mm=DEFAULT_SENSOR_WIDTH_MM):
    """对一帧线性光画面采样颗粒并在映射前叠加，返回线性光结果与中间量。"""
    if isinstance(linear_image, (int, np.integer)):
        linear_image = _demo_image(32, 32)
    base = np.asarray(linear_image, dtype=np.float64)
    field = generate(shot_id, frame_index, base.shape, alpha=alpha,
                     grain_diameter_mm=grain_diameter_mm,
                     sensor_width_mm=sensor_width_mm)
    strength, zone_factors = strength_by_region(
        _luma(base), base_strength=base_strength, zones=zones)
    linear_with_grain = grain_before_tonemap(base, field.grain, strength)
    return {
        "linear": linear_with_grain,
        "grain": field.grain,
        "strength": strength,
        "zone_factors": zone_factors,
        "shot_seed": field.shot_seed,
        "lattice_px": field.lattice_px,
    }


def composite(base_linear, grain=None, strength=1.0):
    """线性光加法合成（不含色调映射；映射在 render 中统一进行）。"""
    if isinstance(base_linear, (int, np.integer)):
        base_linear = _demo_image(32, 32)
        grain = temporal_correlation(0, seed_per_shot(1), base_linear.shape)
    base = np.asarray(base_linear, dtype=np.float64)
    if grain is None:
        grain = np.zeros_like(base)
    return base + np.asarray(grain, dtype=np.float64) * np.asarray(
        strength, dtype=np.float64)


# ---------------------------------------------------------------------------
# 全帧渲染
# ---------------------------------------------------------------------------

def render(shot_id, frame_index=0, linear_image=None,
           alpha=DEFAULT_ALPHA, base_strength=DEFAULT_BASE_STRENGTH,
           zones=DEFAULT_ZONES,
           grain_diameter_mm=DEFAULT_GRAIN_DIAMETER_MM,
           sensor_width_mm=DEFAULT_SENSOR_WIDTH_MM):
    """完整管线：镜头种子 → 时域颗粒 → 分区强度 → 映射前叠加 → 软限 → 色调映射。"""
    if isinstance(shot_id, (int, np.integer)) and linear_image is None:
        linear_image = _demo_image(32, 32)
    if linear_image is None:
        linear_image = _demo_image(32, 32)
    sampled = sample(linear_image, frame_index=frame_index, shot_id=shot_id,
                     alpha=alpha, base_strength=base_strength, zones=zones,
                     grain_diameter_mm=grain_diameter_mm,
                     sensor_width_mm=sensor_width_mm)
    # 软限在映射前收纳 HDR 高光与颗粒，再进 ACES；全程无硬裁切
    softened = soft_limit(sampled["linear"], white=1.0)
    display = _tonemap_aces(softened)
    return {
        "shot_id": shot_id,
        "frame": int(frame_index),
        "display": display,
        "linear_pre_tonemap": sampled["linear"],
        "linear_soft_limited": softened,
        "grain": sampled["grain"],
        "strength": sampled["strength"],
        "zone_factors": sampled["zone_factors"],
        "lattice_px": sampled["lattice_px"],
        "shot_seed": sampled["shot_seed"],
    }


# ---------------------------------------------------------------------------
# 导出 / 指标 / 审计 / 兜底 / 确定性
# ---------------------------------------------------------------------------

def export(frames, path=None):
    """把帧序列序列化为 npy 字节；给定 ``path`` 时写文件并返回路径。"""
    if isinstance(frames, (int, np.integer)):
        frames = [render(1, 0)["display"]]
    if isinstance(frames, dict):
        frames = [frames["display"]]
    stack = np.asarray(list(frames), dtype=np.float64)
    if path is None:
        buf = io.BytesIO()
        np.save(buf, stack)
        return buf.getvalue()
    np.save(path, stack)
    return str(path)


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ac = a - a.mean()
    bc = b - b.mean()
    denom = np.sqrt((ac * ac).sum() * (bc * bc).sum())
    return float((ac * bc).sum() / denom) if denom > 1e-12 else 0.0


def metrics(frames):
    """评估帧序列的时域闪烁、颗粒相关性、明暗强度比与硬裁切比例。

    ``hard_clipped_ratio`` 只统计线性输入超过 1 的 HDR 高光中，
    输出恰好卡在硬边界（0 或 1）的比例——纯黑等合法边界不计入。
    完整管线走软限幅时该值应为 0。
    """
    if isinstance(frames, (int, np.integer)):
        out_seq = [render(1, t) for t in range(6)]
        seq = [o["display"] for o in out_seq]
        pre = [o["linear_pre_tonemap"] for o in out_seq]
    else:
        seq = [np.asarray(f, dtype=np.float64) for f in frames]
        pre = [np.maximum(f, 0.0) for f in seq]
    diffs = [np.abs(seq[t] - seq[t - 1]).mean()
             for t in range(1, len(seq))]
    grain_corr = [_pearson(seq[t - 1], seq[t]) for t in range(1, len(seq))]
    clipped = []
    for disp, linear in zip(seq, pre):
        hdr = linear > 1.0
        if hdr.any():
            clipped.append(np.mean((disp[hdr] <= 0.0) | (disp[hdr] >= 1.0)))
        else:
            clipped.append(0.0)
    hard_clipped = float(np.mean(clipped))
    strength, zone_factors = strength_by_region(_luma(seq[len(seq) // 2]))
    return {
        "frame_count": len(seq),
        "temporal_flicker": float(np.mean(diffs)) if diffs else 0.0,
        "frame_grain_correlation": float(np.mean(grain_corr)) if grain_corr else 0.0,
        "shadow_to_highlight_strength": float(
            zone_factors[0].mean() / max(1e-12, zone_factors[-1].mean())),
        "hard_clipped_ratio": hard_clipped,
        "mean_strength": float(strength.mean()),
    }


def audit(shot_id=1, frame_count=8, shape=DEFAULT_SHAPE):
    """对六项修复逐条做可复现检查，返回每项布尔结论与总判定。"""
    shot_seed = seed_per_shot(shot_id)
    fields = [generate(shot_id, t, shape) for t in range(frame_count)]
    corrs = [_pearson(fields[t - 1].grain, fields[t].grain)
             for t in range(1, frame_count)]
    mean_corr = float(np.mean(corrs))

    # 1) 时域相关：相关系数落在 alpha 附近且明显非零
    temporal_ok = 0.5 < mean_corr < 0.99

    # 2) 分区强度：暗区增益严格大于高光区
    img = _demo_image(*shape)
    _, zone_factors = strength_by_region(_luma(img))
    region_ok = zone_factors[0].mean() > zone_factors[-1].mean()

    # 3) 镜头种子：同镜头同帧逐位一致，不同镜头独立
    g_a = generate(shot_id, 3, shape).grain
    g_b = generate(shot_id, 3, shape).grain
    g_other = generate(shot_id + 1 if isinstance(shot_id, int) else "other",
                       3, shape).grain
    seed_ok = np.array_equal(g_a, g_b) and not np.array_equal(g_a, g_other)

    # 4) 映射前叠加：线性中间量允许 >1，显示结果仍在 [0,1]
    out = render(shot_id, 0, img)
    ordering_ok = (out["linear_pre_tonemap"].max() > 1.0
                   and out["display"].min() >= -1e-12
                   and out["display"].max() <= 1.0 + 1e-12)

    # 5) 物理尺寸：分辨率翻倍，晶格像素间距同比翻倍
    s1 = size_by_physical(1920, 1080)
    s2 = size_by_physical(3840, 2160)
    physical_ok = s2["lattice_px"] == 2 * s1["lattice_px"]

    # 6) 软限幅：无任何像素落在硬边界，且单调
    probe = np.linspace(-4, 4, 1001)
    limited = soft_limit(probe)
    soft_ok = (np.all(limited > -1.0) and np.all(limited < 1.0)
               and np.all(np.diff(limited) > 0))

    checks = {
        "temporal_correlation": temporal_ok,
        "strength_by_region": region_ok,
        "seed_per_shot": seed_ok,
        "grain_before_tonemap": ordering_ok,
        "size_by_physical": physical_ok,
        "soft_limit": soft_ok,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "measured_frame_correlation": mean_corr,
        "target_alpha": DEFAULT_ALPHA,
    }


def fallback(frame_index, shot_seed=None, shape=DEFAULT_SHAPE,
             previous=None, alpha=DEFAULT_ALPHA, lattice_px=4):
    """上一帧缺失时的兜底：无状态回放晶格重建当前帧。

    ``previous`` 给定时视为上一帧的**晶格**（即返回值里的 ``lattice``），
    做单步 AR(1) 递推；缺失时从第 0 帧回放，与 :func:`temporal_correlation`
    走完全相同的回放、上采样与归一化路径，结果逐位一致
    （确定性由镜头种子保证）。
    """
    if isinstance(frame_index, (int, np.integer)) and shot_seed is None:
        shot_seed = seed_per_shot(1)
    shot_seed = int(shot_seed)
    frame_index = int(frame_index)
    lattice_shape = _lattice_shape(shape, max(1, int(lattice_px)))
    innovation_sigma = float(np.sqrt(max(0.0, 1.0 - alpha * alpha)))
    replayed = previous is None
    if replayed:
        lattice = _ar1_lattice(shot_seed, frame_index, lattice_shape, alpha)
    else:
        eps = _frame_rng(shot_seed, frame_index).standard_normal(lattice_shape)
        lattice = alpha * np.asarray(previous) + innovation_sigma * eps
    grain = _upsample_bilinear(lattice, shape)
    std = grain.std()
    if std > 1e-12:
        grain = grain / std
    return {"frame": frame_index, "grain": grain,
            "lattice": lattice, "replayed": replayed}


def determinism(shot_id=1, frame_index=3, shape=DEFAULT_SHAPE):
    """验证同镜头重复渲染逐位一致，且乱序（模拟并行）结果相同。"""
    first = generate(shot_id, frame_index, shape).grain
    second = generate(shot_id, frame_index, shape).grain
    # 模拟并行 worker：先渲染后面的帧再渲染目标帧，随机流互不干扰
    _ = generate(shot_id, frame_index + 5, shape)
    _ = generate(shot_id, 0, shape)
    parallel = generate(shot_id, frame_index, shape).grain
    equal = bool(np.array_equal(first, second) and np.array_equal(first, parallel))
    return {
        "equal": equal,
        "max_abs_diff": float(np.max(np.abs(first - second))),
        "shot_seed": seed_per_shot(shot_id),
        "frame": int(frame_index),
    }
