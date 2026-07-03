#!/usr/bin/env python3
"""CLI: Extract frames from seed video datasets."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extract.frame_extractor import extract_frames


def main():
    parser = argparse.ArgumentParser(description="Extract frames from seed videos")
    parser.add_argument("--video-dir", required=True, help="Directory with video files")
    parser.add_argument("--output-dir", required=True, help="Where to save extracted frames")
    parser.add_argument("--frames-per-cut", type=int, default=1)
    parser.add_argument("--total-cuts", type=int, default=None)
    args = parser.parse_args()

    paths = extract_frames(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        frames_per_cut=args.frames_per_cut,
        total_cuts=args.total_cuts,
    )
    print(f"Done. {len(paths)} frames saved to {args.output_dir}")


if __name__ == "__main__":
    main()
