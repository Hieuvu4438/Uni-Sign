import argparse
import os
import cv2
import glob
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from rtmlib import Wholebody, draw_skeleton

def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    H, W, C = frame.shape
    return keypoints, scores, [W, H]

def process_video(video_path, output_path, wholebody, max_workers=16, overwrite=False):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"file already exists, skip: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {"keypoints": [], "scores": []}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"fail to open: {video_path}")
        return

    vid_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vid_data.append(frame)
    cap.release()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, frame, wholebody) for frame in vid_data]
        for f in tqdm(futures, desc="Processing frames", total=len(vid_data)):
            results.append(f.result())

    for keypoints, scores, w_h in results:
        data['keypoints'].append(keypoints / np.array(w_h)[None, None])
        data['scores'].append(scores)

    with output_path.open('wb') as file:
        pickle.dump(data, file)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", required=True, help="video directory path")
    parser.add_argument("--tgt_dir", required=True, help="pose output root")

    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--backend", default="onnxruntime", choices=["opencv", "onnxruntime", "openvino"])
    parser.add_argument("--openpose_skeleton", action="store_true", help="use openpose format")
    parser.add_argument("--mode", default="lightweight", choices=["performance", "lightweight", "balanced"],)

    parser.add_argument("--video_extensions", nargs='+', default=["mp4", "avi"])
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively discover videos and preserve their relative directories in tgt_dir.")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.tgt_dir, exist_ok=True)

    wholebody = Wholebody(
        to_openpose=args.openpose_skeleton,
        mode=args.mode,
        backend=args.backend,
        device=args.device
    )

    src_dir = Path(args.src_dir)
    tgt_dir = Path(args.tgt_dir)
    video_files = []
    for ext in args.video_extensions:
        ext = ext.lstrip('.')
        pattern = f'**/*.{ext}' if args.recursive else f'*.{ext}'
        video_files.extend(src_dir.glob(pattern))
    video_files = sorted(set(video_files))

    print(f"found {len(video_files)} videos")

    for video_path in tqdm(video_files, desc="Processing"):
        relative_path = video_path.relative_to(src_dir)
        output_path = (tgt_dir / relative_path).with_suffix('.pkl')
        process_video(
            video_path=video_path,
            output_path=output_path,
            wholebody=wholebody,
            max_workers=args.max_workers,
            overwrite=args.overwrite
        )


if __name__ == "__main__":
    main()
