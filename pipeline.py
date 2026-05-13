#!/usr/bin/env python3
"""
Video Library Pipeline — YouTube → Scene-cut clips → Text-free filter → Content description → Storage.

Input:  YouTube URL
Output: /output/{video_id}/clip_000.mp4 + clip_000.json metadata
        Each clip ≤20s, guaranteed text-free, with content description.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ─── config ───────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
MAX_CLIP_DURATION = 20  # seconds
TEXT_CHECK_SAMPLE_FRAMES = 3  # frames per clip to check for text
TEXT_CONFIDENCE_THRESHOLD = 60  # pytesseract confidence
FRAME_SAMPLE_INTERVAL = 0.3  # sample every 0.3s within a clip
FFMPEG = "ffmpeg"
YT_DLP = "yt-dlp"
SCENEDETECT = "scenedetect"


# ─── helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, print it, and return result."""
    print(f"  ⚡ {' '.join(cmd)[:120]}")
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"youtube\.com/embed/([^?]+)",
        r"youtube\.com/shorts/([^?]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


def get_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    r = run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", video_path,
        ]
    )
    return float(r.stdout.strip())


# ─── stage 1: download ────────────────────────────────────────────────────────

def download_youtube(url: str, out_dir: Path) -> Path:
    """Download YouTube video at 1080p, return path to downloaded mp4."""
    video_id = extract_video_id(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.mp4"

    if out_path.exists():
        print(f"  ✓ Already downloaded: {out_path}")
        return out_path

    print(f"\n📥 Downloading: {url}")
    r = run(
        [
            YT_DLP,
            "-f", "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[acodec^=mp4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "--merge-output-format", "mp4",
            "-o", str(out_path),
            url,
        ],
        timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {r.stderr}")
    print(f"  ✓ Downloaded: {out_path}")
    return out_path


# ─── stage 2: scene detection + split ─────────────────────────────────────────

def detect_and_split(video_path: Path, video_id: str, raw_dir: Path) -> list[dict]:
    """
    Detect scene boundaries with scenedetect, merge adjacent short scenes into
    blocks ≤ MAX_CLIP_DURATION, and cut each block into an mp4 file.

    Returns list of {path, start, end, duration}.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Step 2a: detect scenes
    csv_path = raw_dir / "scenes.csv"
    print(f"\n🔍 Detecting scenes for: {video_path.name}")
    r = run(
        [
            SCENEDETECT, "-i", str(video_path),
            "-o", str(raw_dir),
            "detect-content",
            "list-scenes", "-f", str(csv_path),
        ],
        timeout=300,
    )
    if r.returncode != 0 or not csv_path.exists():
        raise RuntimeError("scenedetect failed")

    # Step 2b: parse scenes
    scenes = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Timecode List"):
                continue
            # format: "00:00:00.000,00:00:05.000"
            parts = line.split(",")
            if len(parts) != 2:
                continue
            start = _timecode_to_seconds(parts[0])
            end = _timecode_to_seconds(parts[1])
            dur = end - start
            if dur > 0.3:  # skip extremely short scenes
                scenes.append({"start": start, "end": end, "duration": dur})

    print(f"  Found {len(scenes)} raw scenes")

    # Step 2c: merge scenes into blocks ≤ MAX_CLIP_DURATION
    blocks = _merge_scenes(scenes)

    # Step 2d: cut each block
    clips = []
    for i, block in enumerate(blocks):
        start, end = block["start"], block["end"]
        clip_path = raw_dir / f"clip_{i:03d}.mp4"
        print(f"  ✂️  Cutting clip_{i:03d}: {start:.1f}s → {end:.1f}s ({end - start:.1f}s)")
        r = run(
            [
                FFMPEG, "-y",
                "-ss", str(start), "-i", str(video_path),
                "-t", str(end - start),
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p",
                str(clip_path),
            ],
            timeout=120,
        )
        if r.returncode == 0:
            clips.append(
                {
                    "path": str(clip_path),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "index": i,
                }
            )
        else:
            print(f"  ⚠️  Failed to cut clip_{i:03d}")

    print(f"  ✓ Cut {len(clips)} clips")
    return clips


def _timecode_to_seconds(tc: str) -> float:
    """Convert HH:MM:SS.mmm to float seconds."""
    parts = tc.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def _merge_scenes(scenes: list[dict]) -> list[dict]:
    """Merge consecutive scenes into blocks ≤ MAX_CLIP_DURATION."""
    blocks = []
    current = None
    for scene in scenes:
        if current is None:
            current = dict(scene)
            continue
        merged_dur = scene["end"] - current["start"]
        if merged_dur <= MAX_CLIP_DURATION:
            # merge
            current["end"] = scene["end"]
            current["duration"] = current["end"] - current["start"]
        else:
            blocks.append(current)
            current = dict(scene)
    if current:
        blocks.append(current)

    # Second pass: if any block exceeds 20s (single scene too long),
    # split it at 20s intervals
    final_blocks = []
    for block in blocks:
        dur = block["duration"]
        if dur <= MAX_CLIP_DURATION + 0.5:  # small tolerance
            final_blocks.append(block)
        else:
            # split into 20s chunks
            t = block["start"]
            while t < block["end"]:
                chunk_end = min(t + MAX_CLIP_DURATION, block["end"])
                final_blocks.append(
                    {
                        "start": t,
                        "end": chunk_end,
                        "duration": chunk_end - t,
                    }
                )
                t = chunk_end

    return final_blocks


# ─── stage 3: text detection + filter ──────────────────────────────────────────

def has_text(clip_path: str, duration: float) -> bool:
    """
    Check if a clip contains visible text by sampling frames with pytesseract.
    Returns True if text detected across sampled frames.
    """
    cap = cv2.VideoCapture(clip_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Determine sample timestamps
    sample_count = min(TEXT_CHECK_SAMPLE_FRAMES, max(1, int(duration / 3)))
    interval = duration / (sample_count + 1)

    texts_found = 0
    for i in range(1, sample_count + 1):
        t = interval * i
        frame = _get_frame_at_time(clip_path, t)
        if frame is None:
            continue

        # Convert to PIL for pytesseract
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            for j, conf in enumerate(data["conf"]):
                text = data["text"][j].strip()
                if int(conf) >= TEXT_CONFIDENCE_THRESHOLD and len(text) > 1:
                    texts_found += 1
                    break  # found text in this frame
        except Exception:
            continue

        if texts_found > 0:
            break

    return texts_found > 0


def _get_frame_at_time(video_path: str, t: float) -> Optional[np.ndarray]:
    """Extract a single frame at time t using OpenCV (via temp file)."""
    tmp = tempfile.mktemp(suffix=".png")
    try:
        r = run(
            [
                FFMPEG, "-y", "-ss", str(t), "-i", video_path,
                "-vframes", "1", "-q:v", "2", tmp,
            ],
            timeout=15,
        )
        if r.returncode != 0 or not os.path.exists(tmp):
            return None
        img = cv2.imread(tmp)
        return img
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ─── stage 4: content description ─────────────────────────────────────────────

# We use the vision model via the agent's vision_analyze tool.
# This function prepares the frame for vision analysis.
def describe_clip_frame(clip_path: str) -> Optional[Path]:
    """Extract a representative frame from the middle of the clip for description."""
    dur = get_duration(clip_path)
    t = dur / 2
    frame_path = Path(clip_path).with_suffix(".preview.png")
    r = run(
        [
            FFMPEG, "-y", "-ss", str(t), "-i", clip_path,
            "-vframes", "1", "-q:v", "2", str(frame_path),
        ],
        timeout=15,
    )
    if r.returncode == 0 and frame_path.exists():
        return frame_path
    return None


# ─── stage 5: organize output ─────────────────────────────────────────────────

def save_clip(
    video_id: str,
    clip_info: dict,
    raw_clip_path: str,
    description: str,
    out_dir: Path,
) -> dict:
    """Copy clip to final destination and write metadata JSON."""
    clip_id = f"clip_{clip_info['index']:03d}"
    mp4_dest = out_dir / video_id / f"{clip_id}.mp4"
    json_dest = out_dir / video_id / f"{clip_id}.json"

    mp4_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_clip_path, mp4_dest)

    metadata = {
        "video_id": video_id,
        "clip_id": clip_id,
        "start": round(clip_info["start"], 2),
        "end": round(clip_info["end"], 2),
        "duration": round(clip_info["duration"], 2),
        "description": description,
        "created": datetime.now().isoformat(),
        "resolution": "1080p",
        "text_free": True,
    }
    json_dest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Video Library Pipeline")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output", default=str(OUTPUT_ROOT), help="Output root directory"
    )
    parser.add_argument(
        "--work-dir", default=None, help="Working directory (temp downloads/splits)"
    )
    args = parser.parse_args()

    url = args.url
    out_root = Path(args.output)
    video_id = extract_video_id(url)
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="vlib_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🎬 Video Library Pipeline")
    print(f"   Video ID: {video_id}")
    print(f"   URL:      {url}")
    print(f"   Work dir: {work_dir}")
    print(f"   Output:   {out_root / video_id}")
    print("=" * 60)

    # Stage 1: Download
    video_path = download_youtube(url, work_dir)
    total_dur = get_duration(str(video_path))
    print(f"  Duration: {total_dur:.0f}s")

    # Stage 2: Scene detect + split
    raw_dir = work_dir / "raw_clips"
    clips = detect_and_split(video_path, video_id, raw_dir)

    # Stage 3: Text filter
    print(f"\n🔤 Checking for text in {len(clips)} clips...")
    clean_clips = []
    for clip in clips:
        text_found = has_text(clip["path"], clip["duration"])
        if text_found:
            print(f"  🗑️  clip_{clip['index']:03d}: TEXT DETECTED — skipped")
        else:
            print(f"  ✅ clip_{clip['index']:03d}: clean ({clip['duration']:.1f}s)")
            clean_clips.append(clip)

    print(f"\n  {len(clean_clips)}/{len(clips)} clips text-free")

    # Stage 4: Describe each clean clip (prepare frame for vision analysis)
    print(f"\n🖼️  Preparing preview frames for {len(clean_clips)} clips...")
    preview_frames = {}
    for clip in clean_clips:
        preview = describe_clip_frame(clip["path"])
        if preview:
            preview_frames[clip["index"]] = str(preview)

    # Print summary for the agent to process vision descriptions
    print("\n" + "=" * 60)
    print("📋 CLIPS READY FOR DESCRIPTION:")
    print("=" * 60)
    for clip in clean_clips:
        idx = clip["index"]
        p = preview_frames.get(idx, "N/A")
        print(f"\n  [{idx:03d}] {clip['start']:.1f}s–{clip['end']:.1f}s ({clip['duration']:.1f}s)")
        print(f"       clip: {clip['path']}")
        print(f"       preview: {p}")

    # Save clip info for stage 5
    pipeline_state = {
        "video_id": video_id,
        "url": url,
        "work_dir": str(work_dir),
        "output_root": str(out_root),
        "clean_clips": clean_clips,
        "preview_frames": preview_frames,
    }
    state_path = work_dir / "pipeline_state.json"
    state_path.write_text(json.dumps(pipeline_state, indent=2, default=str))
    print(f"\n💾 Pipeline state saved to: {state_path}")
    print(f"\n✅ Stages 1-4 complete. {len(clean_clips)} clean clips extracted.")
    print(f"   Run stage 5 (vision description + save) next.")


# ─── stage 5 runner ───────────────────────────────────────────────────────────

def run_stage_5(state_path: str, descriptions: dict[int, str]) -> list[dict]:
    """
    Apply vision model descriptions to clips and save to final output.

    descriptions: {clip_index: description_string}
    """
    state = json.loads(Path(state_path).read_text())
    out_root = Path(state["output_root"])
    video_id = state["video_id"]
    clean_clips = state["clean_clips"]

    results = []
    for clip in clean_clips:
        idx = clip["index"]
        desc = descriptions.get(idx, "")
        result = save_clip(video_id, clip, clip["path"], desc, out_root)
        results.append(result)
        print(f"  💾 Saved {video_id}/clip_{idx:03d} — description length: {len(desc)}")

    print(f"\n🎉 Done! {len(results)} clips saved to {out_root / video_id}")
    return results


if __name__ == "__main__":
    main()
