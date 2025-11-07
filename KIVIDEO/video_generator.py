#!/usr/bin/env python3
"""
Local video generation pipeline tailored for Snapdragon X Elite systems.

This script ties together three key stages:
1. Prompt optimisation with Ollama (running locally on http://localhost:11434).
2. Frame generation via an existing ComfyUI workflow (http://localhost:8188).
3. Video compilation from generated frames using FFmpeg with ARM-friendly encoders.

The workflow expects a `workflow_api.json` file compatible with ComfyUI's API.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from ollama import Client  # type: ignore
from websocket import WebSocket, create_connection
import tempfile


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
COMFYUI_WORKFLOW_PATH = Path(
    os.getenv("COMFYUI_WORKFLOW", "workflow_api.json"),
).resolve()
COMFYUI_OUTPUT_ROOT = Path(
    os.getenv("COMFYUI_OUTPUT_ROOT", "./ComfyUI"),
).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./generated_videos")).resolve()
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "24"))
THERMAL_MONITOR_ENABLED = os.getenv("THERMAL_MONITOR_ENABLED", "1") == "1"
THERMAL_MAX_TEMP_C = float(os.getenv("THERMAL_MAX_TEMP_C", "85"))
THERMAL_RESUME_TEMP_C = float(os.getenv("THERMAL_RESUME_TEMP_C", "80"))
THERMAL_POLL_INTERVAL = float(os.getenv("THERMAL_POLL_INTERVAL", "10"))
THERMAL_ZONE_FILTER = [
    name.strip()
    for name in os.getenv("THERMAL_ZONES", "").split(",")
    if name.strip()
]

LOGGER = logging.getLogger("video_generator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

_OLLAMA_CLIENT: Optional[Client] = None
_FFMPEG_ENCODER: Optional[str] = None


class VideoGenerationError(RuntimeError):
    """Raised when the pipeline fails at any stage."""


@dataclass
class SegmentDefinition:
    """Definition for a single segment within a batch run."""

    segment_id: str
    prompt: str
    timespan: Optional[str] = None
    description: Optional[str] = None
    continuity: Optional[str] = None
    seed: Optional[int] = None


def ensure_ffmpeg_available() -> None:
    """Verify that FFmpeg is present on the system."""
    LOGGER.debug("Checking FFmpeg availability")
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        LOGGER.error("FFmpeg is required but not available: %s", exc)
        raise VideoGenerationError("FFmpeg is not installed or not accessible") from exc


def detect_ffmpeg_encoder() -> str:
    """Detect a hardware-friendly FFmpeg encoder, falling back to libx264."""
    global _FFMPEG_ENCODER

    if _FFMPEG_ENCODER:
        return _FFMPEG_ENCODER

    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        encoder_list = result.stdout
        if "hevc_videotoolbox" in encoder_list:
            _FFMPEG_ENCODER = "hevc_videotoolbox"
        else:
            _FFMPEG_ENCODER = "libx264"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        LOGGER.warning("Unable to query FFmpeg encoders (%s); defaulting to libx264", exc)
        _FFMPEG_ENCODER = "libx264"

    LOGGER.debug("Using FFmpeg encoder: %s", _FFMPEG_ENCODER)
    return _FFMPEG_ENCODER


def read_thermal_zones() -> List[Tuple[str, float]]:
    """
    Read available thermal zone temperatures (in Celsius) from sysfs.

    Returns:
        List of tuples containing (zone_name, temperature_celsius).
    """
    thermal_base = Path("/sys/class/thermal")
    if not thermal_base.exists():
        return []

    readings: List[Tuple[str, float]] = []
    for zone in thermal_base.glob("thermal_zone*"):
        try:
            zone_type = (zone / "type").read_text(encoding="utf-8").strip()
        except OSError:
            zone_type = zone.name

        try:
            raw_temp = (zone / "temp").read_text(encoding="utf-8").strip()
        except OSError:
            continue

        try:
            value = float(raw_temp)
        except ValueError:
            continue

        # Values are typically reported in millidegrees Celsius.
        if value > 200:  # heuristic: likely millidegrees
            value /= 1000.0

        readings.append((zone_type, value))

    return readings


def current_cpu_temperature() -> Optional[float]:
    """
    Determine the highest relevant temperature among thermal zones.

    Returns:
        Maximum temperature in Celsius, or None if no sensors are available.
    """
    readings = read_thermal_zones()
    if not readings:
        return None

    if THERMAL_ZONE_FILTER:
        readings = [reading for reading in readings if reading[0] in THERMAL_ZONE_FILTER]
        if not readings:
            return None

    return max(temp for _, temp in readings)


def wait_for_safe_temperature() -> None:
    """
    Pause execution if system temperature exceeds configured thresholds.
    """
    if not THERMAL_MONITOR_ENABLED:
        return

    temperature = current_cpu_temperature()
    if temperature is None:
        LOGGER.debug("No thermal sensors detected; skipping thermal guard")
        return

    if temperature < THERMAL_MAX_TEMP_C:
        return

    LOGGER.warning(
        "System temperature %.1f°C exceeds %.1f°C. Pausing until it cools below %.1f°C",
        temperature,
        THERMAL_MAX_TEMP_C,
        THERMAL_RESUME_TEMP_C,
    )

    while True:
        time.sleep(THERMAL_POLL_INTERVAL)
        temperature = current_cpu_temperature()
        if temperature is None:
            LOGGER.debug("Thermal sensors unavailable during wait; resuming batch")
            break
        if temperature <= THERMAL_RESUME_TEMP_C:
            LOGGER.info("Temperature back to %.1f°C. Resuming batch processing.", temperature)
            break
        LOGGER.debug("Current temperature %.1f°C; waiting %.1f seconds", temperature, THERMAL_POLL_INTERVAL)


def get_ollama_client() -> Client:
    """Create (or reuse) an Ollama client pointing to the configured host."""
    global _OLLAMA_CLIENT

    if _OLLAMA_CLIENT is None:
        LOGGER.debug("Initialising Ollama client at %s", OLLAMA_URL)
        _OLLAMA_CLIENT = Client(host=OLLAMA_URL)

    return _OLLAMA_CLIENT


def optimize_prompt_with_ollama(user_prompt: str) -> str:
    """
    Optimise a raw user prompt using an Ollama language model.

    Args:
        user_prompt: The raw prompt provided by the user.

    Returns:
        A refined prompt string tailored for video generation. If optimisation fails,
        the original prompt is returned as a fallback.
    """
    if not user_prompt.strip():
        raise ValueError("User prompt must not be empty")

    LOGGER.info("Optimising prompt with Ollama model %s", OLLAMA_MODEL)

    try:
        client = get_ollama_client()
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that rewrites prompts for AI video generation. "
                        "Keep the instructions concise, technically precise, and suitable for ComfyUI."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        LOGGER.error("Failed to contact Ollama: %s", exc)
        LOGGER.info("Falling back to original prompt")
        return user_prompt

    message = response.get("message", {})
    optimised_prompt = message.get("content")
    if not optimised_prompt:
        LOGGER.warning("Ollama response missing content; returning original prompt")
        return user_prompt

    LOGGER.debug("Optimised prompt received (%d characters)", len(optimised_prompt))
    return optimised_prompt.strip()


def load_workflow_template(path: Path) -> Dict[str, Any]:
    """
    Load the ComfyUI workflow template from disk.

    Args:
        path: Path to the workflow JSON file.

    Returns:
        Workflow dictionary ready for modification.
    """
    if not path.exists():
        raise VideoGenerationError(f"ComfyUI workflow file not found: {path}")

    LOGGER.debug("Loading ComfyUI workflow from %s", path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def update_workflow_prompt_and_seed(
    workflow: Dict[str, Any],
    prompt: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Replace the positive prompts and random seed within a ComfyUI workflow.

    Args:
        workflow: Parsed workflow dictionary.
        prompt: New positive prompt text.
        seed: Random seed for stochastic nodes.

    Returns:
        Updated workflow dictionary.
    """
    workflow_copy = json.loads(json.dumps(workflow))
    prompt_nodes = 0
    sampler_nodes = 0

    for node in workflow_copy.values():
        if not isinstance(node, dict):
            continue

        class_type = node.get("class_type")
        inputs = node.get("inputs", {})

        if class_type == "CLIPTextEncode" and "text" in inputs:
            inputs["text"] = prompt
            prompt_nodes += 1

        if class_type in {"KSampler", "KSamplerAdvanced", "KSamplerWithControls"}:
            inputs["seed"] = seed
            sampler_nodes += 1

    LOGGER.debug(
        "Updated %d CLIPTextEncode nodes and %d sampler nodes with new prompt and seed",
        prompt_nodes,
        sampler_nodes,
    )

    return workflow_copy


def compose_prompts(
    master_prompt: Optional[str],
    segment_prompt: str,
    metadata: Optional[str] = None,
) -> str:
    """
    Combine master and segment prompts into a single instruction block.

    Args:
        master_prompt: Shared instructions applied to every segment.
        segment_prompt: Segment-specific instructions.
        metadata: Optional contextual metadata (e.g., segment id, timespan).

    Returns:
        Composite prompt string suitable for optimisation.
    """
    parts: List[str] = []

    if master_prompt:
        parts.append(master_prompt.strip())

    if metadata:
        parts.append(metadata.strip())

    parts.append(segment_prompt.strip())

    return "\n\n".join(part for part in parts if part)


def build_ws_url(http_url: str) -> str:
    """
    Derive the ComfyUI WebSocket endpoint from the base HTTP URL.

    Args:
        http_url: Base HTTP URL for ComfyUI (e.g., http://localhost:8188).

    Returns:
        WebSocket URL pointing to the same host/port.
    """
    if http_url.startswith("https://"):
        return http_url.replace("https://", "wss://", 1) + "/ws"
    if http_url.startswith("http://"):
        return http_url.replace("http://", "ws://", 1) + "/ws"
    raise ValueError(f"Unsupported COMFYUI_URL scheme: {http_url}")


def wait_for_comfyui_completion(
    ws: WebSocket,
    prompt_id: str,
    timeout: int = 600,
) -> None:
    """
    Block until ComfyUI finishes executing the queued workflow.

    Args:
        ws: Active WebSocket connection to ComfyUI.
        prompt_id: Identifier of the queued workflow prompt.
        timeout: Maximum time in seconds to wait for completion.
    """
    start_time = time.time()
    ws.settimeout(10)

    LOGGER.info("Waiting for ComfyUI execution to complete (prompt_id=%s)", prompt_id)

    while True:
        if time.time() - start_time > timeout:
            raise VideoGenerationError("ComfyUI execution timed out")

        try:
            message_raw = ws.recv()
        except Exception as exc:
            LOGGER.warning("Transient WebSocket error: %s", exc)
            continue

        try:
            message = json.loads(message_raw)
        except json.JSONDecodeError:
            LOGGER.debug("Ignoring non-JSON WebSocket message: %s", message_raw)
            continue

        message_type = message.get("type")
        data = message.get("data", {})

        if message_type == "execution_error":
            details = data.get("error", "Unknown error")
            raise VideoGenerationError(f"ComfyUI execution error: {details}")

        if message_type == "execution_cancelled":
            raise VideoGenerationError("ComfyUI execution was cancelled")

        if message_type == "executed" and data.get("prompt_id") == prompt_id:
            node_id = data.get("node_id")
            LOGGER.debug("Node %s executed", node_id)
            # Continue until we see the 'execution_end' signal
            continue

        if message_type == "execution_end" and data.get("prompt_id") == prompt_id:
            LOGGER.info("ComfyUI execution finished successfully")
            return


def fetch_comfyui_images(prompt_id: str, frame_dir: Path) -> List[str]:
    """
    Retrieve generated images for a given prompt and copy them into frame_dir.

    Args:
        prompt_id: Identifier of the executed prompt.
        frame_dir: Destination directory for the frames.

    Returns:
        List of absolute file paths to the copied frame images.
    """
    history_url = f"{COMFYUI_URL}/history/{prompt_id}"
    LOGGER.debug("Fetching ComfyUI history from %s", history_url)

    try:
        response = requests.get(history_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise VideoGenerationError(f"Failed to fetch ComfyUI history: {exc}") from exc

    payload = response.json()
    prompt_history = payload.get("history", {}).get(prompt_id)
    if not prompt_history:
        raise VideoGenerationError("ComfyUI history payload is empty")

    outputs = prompt_history.get("outputs", {})
    frames: List[Path] = []

    frame_dir.mkdir(parents=True, exist_ok=True)

    for node_id, node_output in outputs.items():
        images = node_output.get("images", [])
        for image in images:
            filename = image.get("filename")
            subfolder = image.get("subfolder") or image.get("subdirectory") or ""
            image_type = image.get("type", "output")

            if not filename:
                LOGGER.debug("Skipping image without filename in node %s", node_id)
                continue

            source_path = COMFYUI_OUTPUT_ROOT / image_type / subfolder / filename
            if not source_path.exists():
                LOGGER.warning("Expected ComfyUI image missing: %s", source_path)
                continue

            destination = frame_dir / filename
            shutil.copy2(source_path, destination)
            frames.append(destination.resolve())

    if not frames:
        raise VideoGenerationError("No frames were retrieved from ComfyUI outputs")

    frames = sorted(frames, key=natural_sort_key)

    # Honour NUM_FRAMES by trimming or warning
    if len(frames) > NUM_FRAMES:
        LOGGER.info(
            "Trimming frames from %d to %d to honour NUM_FRAMES setting",
            len(frames),
            NUM_FRAMES,
        )
        frames = frames[:NUM_FRAMES]
    elif len(frames) < NUM_FRAMES:
        LOGGER.info(
            "Received %d frame(s), fewer than configured NUM_FRAMES=%d",
            len(frames),
            NUM_FRAMES,
        )

    LOGGER.info("Retrieved %d frame(s) from ComfyUI", len(frames))
    return [str(frame) for frame in frames]


def generate_video_with_comfyui(
    prompt: str,
    output_dir: str = "./output",
    seed: Optional[int] = None,
) -> List[str]:
    """
    Execute the ComfyUI workflow with a new prompt and collect frame outputs.

    Args:
        prompt: Prompt string to inject into the workflow.
        output_dir: Directory to store copied frame images.
        seed: Optional random seed for reproducible generations.

    Returns:
        List of frame file paths generated by the workflow.
    """
    workflow = load_workflow_template(COMFYUI_WORKFLOW_PATH)
    random_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    updated_workflow = update_workflow_prompt_and_seed(workflow, prompt, random_seed)

    client_id = str(uuid.uuid4())
    queue_url = f"{COMFYUI_URL}/prompt"
    payload = {"prompt": updated_workflow, "client_id": client_id}
    LOGGER.info("Queuing workflow at %s (seed=%d)", queue_url, random_seed)

    try:
        response = requests.post(queue_url, json=payload, timeout=30)
    except requests.RequestException as exc:
        raise VideoGenerationError(f"Failed to queue ComfyUI workflow: {exc}") from exc

    if not response.ok:
        detail = response.text
        try:
            detail_json = response.json()
            detail = json.dumps(detail_json, ensure_ascii=False)
        except ValueError:
            pass
        raise VideoGenerationError(
            f"Failed to queue ComfyUI workflow: HTTP {response.status_code} - {detail}",
        )
    response_data = response.json()
    prompt_id = response_data.get("prompt_id")

    if not prompt_id:
        raise VideoGenerationError("ComfyUI did not return a prompt_id")

    ws_url = build_ws_url(COMFYUI_URL)
    LOGGER.debug("Connecting to ComfyUI WebSocket at %s", ws_url)

    try:
        ws = create_connection(ws_url, timeout=10, suppress_origin=True)  # type: ignore[arg-type]
    except Exception as exc:
        raise VideoGenerationError(f"Could not connect to ComfyUI WebSocket: {exc}") from exc

    try:
        wait_for_comfyui_completion(ws, prompt_id)
    finally:
        ws.close()

    frame_output_dir = Path(output_dir).resolve()
    frame_subdir = frame_output_dir / f"frames_{prompt_id}"

    return fetch_comfyui_images(prompt_id, frame_subdir)


def load_segment_definitions(path: Path) -> List[SegmentDefinition]:
    """
    Load segment definitions for batch processing from a JSON file.

    Args:
        path: Path to the JSON document describing the segments.

    Returns:
        List of SegmentDefinition objects.
    """
    if not path.exists():
        raise VideoGenerationError(f"Segment definition file not found: {path}")

    LOGGER.info("Loading segment batch definition from %s", path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise VideoGenerationError("Segment definition file must contain a JSON array")

    segments: List[SegmentDefinition] = []

    for index, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise VideoGenerationError(f"Invalid segment entry at index {index}: {entry}")

        prompt = entry.get("prompt")
        if not prompt:
            raise VideoGenerationError(f"Segment entry {index} missing 'prompt' field")

        segment_id = str(entry.get("id") or f"segment_{index:02d}")

        segments.append(
            SegmentDefinition(
                segment_id=segment_id,
                prompt=str(prompt),
                timespan=entry.get("timespan"),
                description=entry.get("description"),
                continuity=entry.get("continuity"),
                seed=entry.get("seed"),
            ),
        )

    LOGGER.info("Loaded %d segment definition(s)", len(segments))
    return segments


def read_master_prompt(path: Optional[Path]) -> Optional[str]:
    """
    Read a master prompt document if a path is provided.

    Args:
        path: Optional filesystem path.

    Returns:
        String contents of the master prompt, or None if no path supplied.
    """
    if path is None:
        return None

    if not path.exists():
        raise VideoGenerationError(f"Master prompt file not found: {path}")

    LOGGER.info("Loading master prompt from %s", path)
    return path.read_text(encoding="utf-8")


def sanitize_segment_id(raw_id: str) -> str:
    """
    Create a filesystem-friendly identifier for segment output paths.

    Args:
        raw_id: Original identifier from the segment definition.

    Returns:
        Sanitized identifier limited to alphanumeric, dash, and underscore.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_id).strip("_")
    return sanitized or "segment"


def generate_batch(
    segments_file: Path,
    master_prompt_file: Optional[Path] = None,
    base_seed: Optional[int] = None,
    concat_output: Optional[Path] = None,
) -> List[str]:
    """
    Generate multiple segment videos sequentially using predefined prompts.

    Args:
        segments_file: JSON file enumerating segment prompts and metadata.
        master_prompt_file: Optional shared instructions applied to every segment.
        base_seed: Base seed used when a segment definition omits an explicit seed.
        concat_output: Optional path for concatenated final video.

    Returns:
        List of absolute paths to the generated segment video files.
    """
    segments = load_segment_definitions(segments_file)
    master_prompt = read_master_prompt(master_prompt_file)

    if base_seed is None:
        base_seed = random.randint(0, 2**31 - 1)
        LOGGER.info("No base seed supplied; using generated base seed %d", base_seed)
    else:
        LOGGER.info("Using provided base seed %d", base_seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_videos: List[str] = []
    start_time = time.time()

    for idx, segment in enumerate(segments):
        segment_seed = segment.seed if segment.seed is not None else base_seed + idx
        metadata_lines: List[str] = [f"SEGMENT {segment.segment_id}"]

        if segment.timespan:
            metadata_lines[-1] += f" ({segment.timespan})"

        if segment.description:
            metadata_lines.append(f"Beschreibung: {segment.description}")

        if segment.continuity:
            metadata_lines.append(f"Kontinuität: {segment.continuity}")

        metadata = "\n".join(metadata_lines)
        output_name = f"{sanitize_segment_id(segment.segment_id)}.mp4"

        wait_for_safe_temperature()

        LOGGER.info(
            "Processing %s with seed %d -> %s",
            segment.segment_id,
            segment_seed,
            output_name,
        )

        video_path = main(
            segment.prompt,
            output_filename=output_name,
            seed=segment_seed,
            master_prompt=master_prompt,
            metadata=metadata,
        )
        generated_videos.append(video_path)

    elapsed = time.time() - start_time
    LOGGER.info("Batch generation finished in %.2f seconds", elapsed)

    if concat_output and generated_videos:
        concat_path = (
            concat_output
            if concat_output.is_absolute()
            else (OUTPUT_DIR / concat_output).resolve()
        )
        success = concat_videos(generated_videos, concat_path)
        if not success:
            raise VideoGenerationError("Concatenation of segment videos failed")
        generated_videos.append(str(concat_path.resolve()))

    return generated_videos


def concat_videos(video_paths: Sequence[str], output_path: Path) -> bool:
    """
    Concatenate multiple MP4 segments into a single video using FFmpeg.

    Args:
        video_paths: Ordered collection of video file paths.
        output_path: Target file for the concatenated video.

    Returns:
        True if concatenation succeeds, otherwise False.
    """
    ensure_ffmpeg_available()

    if not video_paths:
        LOGGER.error("No videos provided for concatenation")
        return False

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    list_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as list_file:
            for video in video_paths:
                list_file.write(f"file '{Path(video).resolve()}'\n")
            list_path = Path(list_file.name)
    except OSError as exc:
        LOGGER.error("Failed to prepare FFmpeg concat list: %s", exc)
        return False

    if list_path is None:
        LOGGER.error("Concat list was not created")
        return False

    encoder = detect_ffmpeg_encoder()
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    LOGGER.info("Concatenating %d videos into %s", len(video_paths), output_path)

    try:
        subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        LOGGER.error(
            "FFmpeg concat failed: %s",
            exc.stderr.decode("utf-8", errors="ignore"),
        )
        return False
    finally:
        if list_path is not None:
            try:
                list_path.unlink()
            except OSError:
                LOGGER.debug("Temporary concat list already removed")

    LOGGER.info("Concatenated video written to %s", output_path)
    return True


def natural_sort_key(path: Path) -> List[Any]:
    """
    Produce a sort key that honours numeric substrings within filenames.

    Args:
        path: Path object representing a frame.

    Returns:
        List containing strings/integers to allow natural sorting.
    """
    pattern = re.compile(r"(\d+)")
    parts: List[Any] = []
    for segment in pattern.split(path.name):
        if segment.isdigit():
            parts.append(int(segment))
        else:
            parts.append(segment.lower())
    return parts


def compile_frames_to_video(
    frame_dir: str,
    output_file: str = "output.mp4",
    fps: int = 30,
) -> bool:
    """
    Assemble PNG frames into a 1080p MP4 video using FFmpeg.

    Args:
        frame_dir: Directory containing PNG frames.
        output_file: Destination video filename.
        fps: Output frame rate.

    Returns:
        True if FFmpeg completes successfully, otherwise False.
    """
    ensure_ffmpeg_available()

    frame_path = Path(frame_dir).resolve()
    if not frame_path.exists():
        LOGGER.error("Frame directory does not exist: %s", frame_path)
        return False

    frames = sorted(
        [p for p in frame_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=natural_sort_key,
    )

    if not frames:
        LOGGER.error("No image frames found in %s", frame_path)
        return False

    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = detect_ffmpeg_encoder()
    input_pattern = "\n".join(f"file '{frame}'" for frame in frames) + "\n"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "-",
        "-r",
        str(fps),
        "-vf",
        "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    LOGGER.info("Compiling %d frames into %s", len(frames), output_path)

    try:
        process = subprocess.run(
            ffmpeg_cmd,
            input=input_pattern.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        if process.stderr:
            LOGGER.debug("FFmpeg stderr: %s", process.stderr.decode("utf-8", errors="ignore"))
    except subprocess.CalledProcessError as exc:
        LOGGER.error("FFmpeg reported an error: %s", exc.stderr.decode("utf-8", errors="ignore"))
        return False

    LOGGER.info("Video written to %s", output_path)
    return True


def main(
    user_input: str,
    output_filename: str = "generated_video.mp4",
    seed: Optional[int] = None,
    master_prompt: Optional[str] = None,
    metadata: Optional[str] = None,
) -> str:
    """
    Execute the complete video generation pipeline from prompt to MP4.

    Args:
        user_input: Free-form prompt supplied by the user.
        output_filename: Name for the final video file.
        seed: Optional deterministic seed for the workflow.
        master_prompt: Optional shared instructions prepended to `user_input`.
        metadata: Optional contextual metadata appended between master and segment prompts.

    Returns:
        String path to the generated video file.
    """
    LOGGER.info("Starting video generation workflow")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wait_for_safe_temperature()

    composed_prompt = compose_prompts(master_prompt, user_input, metadata=metadata)

    optimized_prompt = optimize_prompt_with_ollama(composed_prompt)
    frames = generate_video_with_comfyui(
        optimized_prompt,
        output_dir=str(OUTPUT_DIR),
        seed=seed,
    )

    frames_dir = Path(frames[0]).resolve().parent
    output_path = OUTPUT_DIR / output_filename

    success = compile_frames_to_video(str(frames_dir), str(output_path))
    if not success:
        raise VideoGenerationError("Failed to compile frames into video")

    LOGGER.info("Video generation completed: %s", output_path)
    return str(output_path)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Local KI Video Generator")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Freitext-Prompt für die Videoerstellung",
    )
    parser.add_argument(
        "--batch",
        help="Pfad zu einer JSON-Datei mit mehreren Segment-Prompts",
    )
    parser.add_argument(
        "--master-prompt",
        help="Optionaler Pfad zu einem Master-Prompt (z. B. docs/Veo31MasterPrompt.md)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        help="Basis-Seed für reproduzierbare Ergebnisse (Segment-Seed = base_seed + idx)",
    )
    parser.add_argument(
        "--output",
        default="generated_video.mp4",
        help="Name der Ausgabedatei (MP4)",
    )
    parser.add_argument(
        "--concat-output",
        help="Optionaler Dateiname für ein zusammengefügtes Gesamtvideo",
    )
    args = parser.parse_args()

    try:
        if args.batch:
            master_path = Path(args.master_prompt).resolve() if args.master_prompt else None
            concat_path = Path(args.concat_output).resolve() if args.concat_output else None
            videos = generate_batch(
                Path(args.batch).resolve(),
                master_prompt_file=master_path,
                base_seed=args.base_seed,
                concat_output=concat_path,
            )
            print("Erzeugte Dateien:")
            for video in videos:
                print(video)
        else:
            if not args.prompt:
                parser.error("Entweder ein Prompt oder --batch muss angegeben werden.")

            master_prompt_text = read_master_prompt(
                Path(args.master_prompt).resolve(),
            ) if args.master_prompt else None

            result_path = main(
                args.prompt,
                output_filename=args.output,
                seed=args.base_seed,
                master_prompt=master_prompt_text,
            )
            print(f"Video erstellt: {result_path}")
    except Exception as exc:  # pragma: no cover - CLI reporting
        LOGGER.error("Videoerstellung fehlgeschlagen: %s", exc)
        sys.exit(1)
