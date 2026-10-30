"""实验参数配置。

这里只放会直接影响测量结果的物理参数。修改后需要重启 Flask 服务，
Android 端不需要重新安装。
"""

# 拉伸前（T0 阶段）的标距长度，单位 mm。
#
# 速度按“拉伸比率变化量 × 标距长度 ÷ 总时间”计算，再换算为 mm/min：
#     v = (α2 - α0) × L0 ÷ (ΔT1 + ΔT2) × 60
#
# 如果老师给出的试样长度不是 50 mm，只改这一处即可。
PRESET_MATERIAL_LENGTH_MM = 50.0


def resolve_material_length_mm(value=None):
    """返回本次计算使用的材料长度。

    :param value: 调用方显式传入的长度（mm）；None 表示使用上方预设值。
    :raises ValueError: 长度不是正数或无法转换为浮点数。
    """
    candidate = PRESET_MATERIAL_LENGTH_MM if value is None else value
    try:
        length = float(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError("material length must be a positive number") from exc
    if length <= 0:
        raise ValueError("material length must be a positive number")
    return length
