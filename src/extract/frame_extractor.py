"""Extract key frames from seed video datasets for ControlNet conditioning."""

import json
from pathlib import Path

import cv2


def extract_frames(
    video_dir: str,
    output_dir: str,
    frames_per_cut: int = 1,
    total_cuts: int | None = None,
) -> list[Path]:
    """Extract frames from video files in a directory.

    Args:
        video_dir: Directory containing video files.
        output_dir: Where to save extracted frames.
        frames_per_cut: Number of frames to sample per cut/segment.
        total_cuts: If set, segment each video into this many cuts.

    Returns:
        List of paths to extracted frame images.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
    video_files = sorted([
        f for f in video_dir.iterdir()
        if f.suffix.lower() in video_exts
    ])

    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    # Load progress
    progress_file = output_dir / "extract_progress.json"
    completed = set()
    if progress_file.exists():
        completed = set(json.loads(progress_file.read_text()).get("completed", []))

    index = []
    frame_paths = []

    for video_path in video_files:
        if str(video_path) in completed:
            continue

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        if total_frames <= 0:
            cap.release()
            continue

        # Determine cuts
        n_cuts = total_cuts or max(1, total_frames // int(fps * 10))  # default: 10-second segments
        frames_per_video = n_cuts * frames_per_cut

        # Sample frame indices uniformly
        if frames_per_video >= total_frames:
            sample_indices = list(range(0, total_frames, max(1, total_frames // frames_per_video)))
        else:
            cut_size = total_frames // n_cuts
            sample_indices = []
            for c in range(n_cuts):
                cut_start = c * cut_size
                step = cut_size // (frames_per_cut + 1)
                for f in range(frames_per_cut):
                    idx = cut_start + step * (f + 1)
                    sample_indices.append(min(idx, total_frames - 1))

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            fname = f"{video_path.stem}_f{frame_idx:08d}.png"
            out_path = output_dir / fname
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(out_path)

            index.append({
                "frame_path": str(out_path),
                "source_video": str(video_path),
                "frame_index": frame_idx,
                "timestamp_sec": frame_idx / fps,
            })

        cap.release()
        completed.add(str(video_path))

        # Save progress
        progress_file.write_text(json.dumps({"completed": sorted(completed)}))

    # Save index
    index_path = output_dir / "frame_index.jsonl"
    with open(index_path, "a") as f:
        for entry in index:
            f.write(json.dumps(entry) + "\n")

    print(f"Extracted {len(frame_paths)} frames from {len(video_files)} videos")
    return frame_paths
