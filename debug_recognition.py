"""面向非开发人员的单图/视频识别诊断工具。"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="保存识别的每一步图片；输入视频时可检查全部帧或指定帧。"
    )
    parser.add_argument("input", help="图片或视频文件路径")
    parser.add_argument("--frame", type=int, help="只详细检查视频的第几帧（从 1 开始）")
    parser.add_argument("--step", type=int, default=1, help="视频汇总时每隔几帧检查一次，默认 1")
    parser.add_argument("--output", help="输出目录；默认放在 debug_out 下")
    parser.add_argument("--open-folder", action="store_true", help="完成后打开输出目录")

    parser.add_argument("--gamma", type=float, default=0.7, help="亮区增强指数，默认 0.7")
    parser.add_argument("--brightness", type=float, default=120, help="亮度增强起点，默认 120")
    parser.add_argument("--canny-low", type=float, default=50, help="边缘检测低阈值，默认 50")
    parser.add_argument("--canny-high", type=float, default=150, help="边缘检测高阈值，默认 150")
    parser.add_argument("--gaussian", type=int, default=9, help="高斯模糊核（正奇数），默认 9")
    parser.add_argument("--kernel", type=int, default=9, help="闭运算核（正整数），默认 9")
    parser.add_argument("--black", type=float, default=0.15, help="黑色亮度阈值 0~1，默认 0.15")
    parser.add_argument("--center", type=float, default=2 / 3, help="色块中央取色比例，默认 0.667")
    parser.add_argument("--min-area", type=float, default=200, help="最小轮廓面积，默认 200")
    parser.add_argument("--stretch-tolerance", type=float, default=0.3, help="色块长宽比离散容忍度，默认 0.3")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    path = Path(args.input).expanduser()
    if not path.is_file():
        raise SystemExit(f"找不到输入文件：{path}")
    if args.frame is not None and args.frame < 1:
        raise SystemExit("--frame 必须从 1 开始。")
    if args.step < 1:
        raise SystemExit("--step 必须大于等于 1。")
    if args.gaussian < 1 or args.gaussian % 2 == 0:
        raise SystemExit("--gaussian 必须是正奇数，例如 3、5、7、9。")
    if args.kernel < 1:
        raise SystemExit("--kernel 必须是正整数。")
    if not 0 <= args.black <= 1:
        raise SystemExit("--black 必须在 0~1 之间。")
    if not 0 < args.center <= 1:
        raise SystemExit("--center 必须大于 0 且不超过 1。")


def make_output_dir(args: argparse.Namespace, input_path: Path) -> Path:
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("debug_out") / f"{input_path.stem}_{stamp}"
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def detector_for(image, args: argparse.Namespace, detail_dir: Path | None = None):
    detector = ColorCodeDetector(image, pathSwtich=False)
    detector.debug_output_dir = str(detail_dir) if detail_dir else None
    detector.show_details = True
    detector.HP_gamma = args.gamma
    detector.HPbrightness_threshold = args.brightness
    detector.HPt1Canny = args.canny_low
    detector.HPt2Canny = args.canny_high
    detector.HPsizeGaussian = args.gaussian
    detector.HPkernel = args.kernel
    detector.HP_Black_th = args.black
    detector.center_factor = args.center
    detector.min_contour_area = args.min_area
    detector.HP_ts_radio = args.stretch_tolerance
    return detector


def compact_result(result: dict, frame_number: int | None = None, timestamp: float | None = None):
    data = {
        "frame_number": frame_number,
        "timestamp_s": None if timestamp is None else round(timestamp, 4),
        "status": result.get("Status"),
        "error_info": result.get("Error_info"),
        "block_count": result.get("Block_Counts"),
        "stretch_ratio": None,
        "color_matrix": result.get("color_matrix"),
    }
    ratio = result.get("stretch_ratio")
    if ratio is not None:
        data["stretch_ratio"] = round(float(ratio), 6)
    return data


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_image(path: Path, image) -> None:
    """可靠地向 Windows 中文路径写图。"""
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def read_image(path: Path):
    """可靠地从 Windows 中文路径读图。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def analyze_detail(image, args: argparse.Namespace, output_dir: Path, frame_number=None, timestamp=None):
    detail_dir = output_dir / "处理步骤"
    detail_dir.mkdir(parents=True, exist_ok=True)
    write_image(detail_dir / "00_输入原图.png", image)
    detector = detector_for(image, args, detail_dir)
    result = detector.analyze()
    if result.get("pic_toSave") is not None:
        write_image(output_dir / "识别结果总览.png", result["pic_toSave"])
    compact = compact_result(result, frame_number, timestamp)
    save_json(output_dir / "识别结果.json", compact)
    return compact


def read_video_frame(cap, frame_number: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"无法读取第 {frame_number} 帧，请检查帧号是否超过视频长度。")
    return frame


def normalize_video_frame(frame, rotate: bool):
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if rotate else frame


def analyze_video(input_path: Path, args: argparse.Namespace, output_dir: Path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotate = width > height
    metadata = {
        "input": str(input_path.resolve()),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_total,
        "production_rotation_applied": rotate,
    }
    save_json(output_dir / "视频信息.json", metadata)

    if args.frame is not None:
        frame = normalize_video_frame(read_video_frame(cap, args.frame), rotate)
        timestamp = (args.frame - 1) / fps if fps else None
        result = analyze_detail(frame, args, output_dir, args.frame, timestamp)
        cap.release()
        return [result]

    frames_dir = output_dir / "逐帧识别总览"
    frames_dir.mkdir(parents=True, exist_ok=True)
    results = []
    frame_number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        if (frame_number - 1) % args.step:
            continue
        frame = normalize_video_frame(frame, rotate)
        result = detector_for(frame, args).analyze()
        timestamp = (frame_number - 1) / fps if fps else None
        compact = compact_result(result, frame_number, timestamp)
        results.append(compact)
        overview = result.get("pic_toSave")
        if overview is not None:
            cv2.putText(
                overview,
                f"Frame {frame_number}  Time {timestamp or 0:.3f}s",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            write_image(frames_dir / f"frame_{frame_number:06d}.jpg", overview)
        print(f"已处理 {frame_number}/{frame_total} 帧", end="\r", flush=True)
    cap.release()
    print()
    save_json(output_dir / "逐帧识别结果.json", results)
    with (output_dir / "逐帧识别结果.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_number", "timestamp_s", "status", "error_info", "block_count", "stretch_ratio", "color_matrix"],
        )
        writer.writeheader()
        for row in results:
            csv_row = row.copy()
            csv_row["color_matrix"] = json.dumps(row["color_matrix"], ensure_ascii=False)
            writer.writerow(csv_row)
    return results


def parameter_snapshot(args: argparse.Namespace):
    return {
        "gamma": args.gamma,
        "brightness": args.brightness,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "gaussian": args.gaussian,
        "kernel": args.kernel,
        "black": args.black,
        "center": args.center,
        "min_area": args.min_area,
        "stretch_tolerance": args.stretch_tolerance,
        "video_step": args.step,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = make_output_dir(args, input_path)
    save_json(output_dir / "本次参数.json", parameter_snapshot(args))

    suffix = input_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        image = read_image(input_path)
        if image is None:
            raise SystemExit(f"无法读取图片：{input_path}")
        analyze_detail(image, args, output_dir)
    elif suffix in VIDEO_SUFFIXES:
        analyze_video(input_path, args, output_dir)
    else:
        raise SystemExit(f"不支持的文件格式：{suffix}")

    print(f"\n完成。结果保存在：{output_dir}")
    if args.open_folder and os.name == "nt":
        os.startfile(output_dir)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
