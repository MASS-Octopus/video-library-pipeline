# Video Library Pipeline

Tự động tải video YouTube, cắt theo chuyển cảnh, lọc text-free, và tạo mô tả nội dung cho từng clip.

## Tính năng

- **Download** video YouTube 1080p (avc1 + mp4a)
- **Scene detection** — phát hiện ranh giới chuyển cảnh bằng `scenedetect`
- **Merge & split** — gộp các phân cảnh ngắn thành block ≤ 20 giây
- **Text filter** — dùng `pytesseract` OCR để phát hiện và loại bỏ clip có text
- **Content description** — trích frame + vision model mô tả nội dung clip
- **Organized output** — mỗi clip lưu dạng `output/{video_id}/clip_000.mp4` + `clip_000.json`

## Architecture

```
YouTube URL → yt-dlp download → scenedetect cắt cảnh
→ pytesseract text-check từng block → lọc block không có text
→ Vision model mô tả nội dung → lưu clip + metadata JSON
```

## Requirements

- Python 3.10+
- FFmpeg
- Tesseract OCR
- yt-dlp
- scenedetect
- deno (for yt-dlp JS runtime)

## Quick Start

```bash
# Cài dependencies
pip install pytesseract opencv-python-headless Pillow

# Chạy pipeline (stages 1-4: download → cắt → lọc text)
python3 pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Kết quả lưu tại output/{video_id}/
#   clip_000.mp4  ← video sạch text, ≤20s
#   clip_000.json ← metadata: start, end, duration, description, text_free
```

## Output Format

```json
{
  "video_id": "dQw4w9WgXcQ",
  "clip_id": "clip_000",
  "start": 12.5,
  "end": 32.3,
  "duration": 19.8,
  "description": "A person walking through a busy city street...",
  "created": "2026-05-13T13:15:00",
  "resolution": "1080p",
  "text_free": true
}
```

## License

MIT
