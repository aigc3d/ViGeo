from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import zipfile

import cv2
import numpy as np


SEVEN_SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
SEVEN_SCENES_BASE_URL = (
    "https://download.microsoft.com/download/2/8/5/"
    "28564B23-0828-408F-8631-23B1EFF1DAC8"
)
NRGBD_URL = "https://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip"

DEPTH_TO_RGB = np.asarray(
    [
        [9.9996518012567637e-01, 2.6765126468950343e-03, -7.9041012313000904e-03, -2.5558943178152542e-02],
        [-2.7409311281316700e-03, 9.9996302803027592e-01, -8.1504520778013286e-03, 1.0109636268061706e-04],
        [7.8819942130445332e-03, 8.1718328771890631e-03, 9.9993554558014031e-01, 2.0318321729487039e-03],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _download(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / url.rsplit("/", 1)[-1]
    subprocess.run(["wget", "-c", "-O", str(output), url], check=True)
    return output


def download_archives(data_root: Path) -> tuple[list[Path], Path]:
    download_root = data_root / "_downloads"
    seven_archives = [
        _download(f"{SEVEN_SCENES_BASE_URL}/{scene}.zip", download_root / "7scenes")
        for scene in SEVEN_SCENES
    ]
    nrgbd_archive = _download(NRGBD_URL, download_root / "neural_rgbd")
    return seven_archives, nrgbd_archive


def _safe_extract_zip(archive: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            member_path = Path(member.filename)
            if member_path.name.lower() == "thumbs.db" or "__MACOSX" in member_path.parts:
                continue
            target = (output_dir / member.filename).resolve()
            if not str(target).startswith(str(output_dir)):
                raise RuntimeError(f"Refusing to extract suspicious zip member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)


def extract_seven_scenes(archives: list[Path], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for archive, scene in zip(archives, SEVEN_SCENES):
        scene_dir = output_root / scene
        if not (scene_dir / "TestSplit.txt").exists():
            print(f"Extracting {archive.name} ...")
            _safe_extract_zip(archive, output_root)
        nested_archives = sorted(scene_dir.glob("*.zip"))
        for nested in nested_archives:
            sequence_dir = scene_dir / nested.stem
            if (
                sequence_dir.is_dir()
                and any(sequence_dir.glob("frame-*.color.png"))
                and any(sequence_dir.glob("frame-*.depth.png"))
                and any(sequence_dir.glob("frame-*.pose.txt"))
            ):
                continue
            print(f"Extracting {scene}/{nested.name} ...")
            _safe_extract_zip(nested, scene_dir)


def _find_nrgbd_source(extract_root: Path) -> Path:
    if any((extract_root / child).is_dir() and (extract_root / child / "poses.txt").exists() for child in extract_root.iterdir()):
        return extract_root
    for candidate in extract_root.rglob("*"):
        if candidate.is_dir() and any(
            child.is_dir() and (child / "poses.txt").exists() for child in candidate.iterdir()
        ):
            return candidate
    raise RuntimeError(f"Could not locate extracted NRGBD scenes under {extract_root}")


def extract_nrgbd(archive: Path, output_root: Path) -> None:
    if output_root.is_dir() and any((path / "poses.txt").exists() for path in output_root.iterdir() if path.is_dir()):
        return
    extract_root = output_root.parent / "_nrgbd_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name} ...")
    _safe_extract_zip(archive, extract_root)
    source = _find_nrgbd_source(extract_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for scene in source.iterdir():
        if scene.is_dir() and (scene / "poses.txt").exists():
            target = output_root / scene.name
            if not target.exists():
                shutil.move(str(scene), str(target))


def project_depth_to_rgb(raw_depth: np.ndarray) -> np.ndarray:
    depth = raw_depth.astype(np.float64) / 1000.0
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    valid = (depth > 0) & (depth < 100)
    z = depth[valid]
    x = ((xx[valid] + 0.5 - width / 2) / 585.0) * z
    y = ((yy[valid] + 0.5 - height / 2) / 585.0) * z
    points = np.stack((x, y, z, np.ones_like(z)), axis=0)
    points = DEPTH_TO_RGB @ points
    projected_z = points[2]
    projected_x = np.rint(points[0] / projected_z * 525.0 + 320.0).astype(np.int64)
    projected_y = np.rint(points[1] / projected_z * 525.0 + 240.0).astype(np.int64)
    inside = (projected_x >= 0) & (projected_x < 640) & (projected_y >= 0) & (projected_y < 480)

    registered = np.full(480 * 640, 2000.0, dtype=np.float64)
    flat_indices = projected_y[inside] * 640 + projected_x[inside]
    np.minimum.at(registered, flat_indices, projected_z[inside])
    registered[registered > 1000.0] = 0.0
    return (registered.reshape(480, 640) * 1000.0).astype(np.uint16)


def _test_sequence_dirs(scene_dir: Path) -> list[Path]:
    split = scene_dir / "TestSplit.txt"
    sequences = []
    for line in split.read_text().splitlines():
        number = "".join(filter(str.isdigit, line))
        sequences.append(scene_dir / f"seq-{number.zfill(2)}")
    return sequences


def preprocess_seven_scenes(output_root: Path, stride: int = 200, all_frames: bool = False) -> int:
    written = 0
    for scene in SEVEN_SCENES:
        for sequence_dir in _test_sequence_dirs(output_root / scene):
            color_files = sorted(sequence_dir.glob("frame-*.color.png"))
            selected = color_files if all_frames else color_files[::stride]
            for color_path in selected:
                prefix = color_path.name.removesuffix(".color.png")
                raw_path = sequence_dir / f"{prefix}.depth.png"
                projected_path = sequence_dir / f"{prefix}.depth.proj.png"
                if projected_path.exists():
                    continue
                raw_depth = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
                if raw_depth is None:
                    raise FileNotFoundError(f"Missing raw 7-Scenes depth: {raw_path}")
                projected = project_depth_to_rgb(raw_depth)
                if not cv2.imwrite(str(projected_path), projected):
                    raise IOError(f"Failed to write projected depth: {projected_path}")
                written += 1
                print(f"Projected {projected_path}")
    return written


def validate(data_root: Path, stride: int) -> None:
    seven_root = data_root / "7scenes"
    sequence_count = frame_count = 0
    for scene in SEVEN_SCENES:
        split_path = seven_root / scene / "TestSplit.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing 7-Scenes split file: {split_path}")
        for sequence_dir in _test_sequence_dirs(seven_root / scene):
            sequence_count += 1
            selected = sorted(sequence_dir.glob("frame-*.color.png"))[::stride]
            for color_path in selected:
                prefix = color_path.name.removesuffix(".color.png")
                if not (sequence_dir / f"{prefix}.depth.proj.png").exists():
                    raise FileNotFoundError(f"Missing projected depth for {color_path}")
                frame_count += 1
    nrgbd_root = data_root / "neural_rgbd"
    nrgbd_scenes = [path for path in nrgbd_root.iterdir() if path.is_dir() and (path / "poses.txt").exists()]
    if not nrgbd_scenes:
        raise RuntimeError(f"No NRGBD scenes found under {nrgbd_root}")
    print(
        f"Validated 7-Scenes: {sequence_count} test sequences, {frame_count} sampled frames; "
        f"NRGBD: {len(nrgbd_scenes)} scenes."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare 7-Scenes and Neural RGBD for reconstruction evaluation.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--all-frames", action="store_true", help="Project all 7-Scenes depth frames, not only stride-200 evaluation frames.")
    parser.add_argument("--seven-scenes-stride", type=int, default=200)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    download_root = data_root / "_downloads"
    archives = [download_root / "7scenes" / f"{scene}.zip" for scene in SEVEN_SCENES]
    nrgbd_archive = download_root / "neural_rgbd" / "neural_rgbd_data.zip"
    if args.download:
        archives, nrgbd_archive = download_archives(data_root)
    missing = [path for path in [*archives, nrgbd_archive] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing archives: {missing}. Re-run with --download.")
    if args.extract or args.download:
        extract_seven_scenes(archives, data_root / "7scenes")
        extract_nrgbd(nrgbd_archive, data_root / "neural_rgbd")
    written = preprocess_seven_scenes(
        data_root / "7scenes",
        stride=args.seven_scenes_stride,
        all_frames=args.all_frames,
    )
    print(f"Generated {written} new projected 7-Scenes depth maps.")
    validate(data_root, args.seven_scenes_stride)


if __name__ == "__main__":
    main(parse_args())
