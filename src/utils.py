from __future__ import annotations

from pathlib import Path
from typing import List, Dict


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def collect_audio_records(audio_dir: Path) -> List[Dict[str, str]]:
    """
    Collect audio files recursively and infer labels from folder names.
    If a file is directly inside data/audio/, label is set to 'unlabeled'.
    """
    records = []

    for path in sorted(audio_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            rel = path.relative_to(audio_dir)

            if len(rel.parts) >= 2:
                label = rel.parts[0]
            else:
                label = "unlabeled"

            records.append(
                {
                    "file_path": str(path),
                    "original_file_name": path.name,
                    "file_stem": path.stem,
                    "label": label,
                }
            )

    return records


def make_clean_label(file_stem: str, max_len: int = 40) -> str:
    name = file_stem.replace("_", " ").strip()
    return name[:max_len] + "..." if len(name) > max_len else name