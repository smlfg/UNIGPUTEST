import os
import json
import logging
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional

import ollama
import requests
import websocket

# --- Configuration ---
OLLAMA_MODEL = "llama3.2:3b"
COMFYUI_URL = "http://localhost:8188"
OLLAMA_URL = "http://localhost:11434"
OUTPUT_DIR = "./generated_videos"
NUM_FRAMES = 24

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("video_generator.log"),
    ],
)


def check_ffmpeg() -> bool:
    """Checks if FFmpeg is installed and available in the system's PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        logging.info("FFmpeg is installed.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("FFmpeg is not installed. Please install it to continue.")
        return False


def optimize_prompt_with_ollama(user_prompt: str) -> str:
    """
    Optimizes a user's text prompt for video generation using an Ollama model.

    Args:
        user_prompt: The initial text prompt from the user.

    Returns:
        An optimized and more detailed prompt for video generation.
    """
    logging.info(f"Optimizing prompt with Ollama model: {OLLAMA_MODEL}")
    try:
        client = ollama.Client(host=OLLAMA_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in creating highly detailed and effective prompts for AI video generation. Your task is to expand the user's input into a rich, descriptive prompt that specifies the subject, action, setting, style, and composition. The prompt should be a single, comma-separated string.",
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        optimized_prompt = response["message"]["content"]
        logging.info(f"Optimized prompt: {optimized_prompt}")
        return optimized_prompt
    except Exception as e:
        logging.error(f"Error connecting to Ollama: {e}")
        # Fallback to user prompt if Ollama fails
        return user_prompt


def generate_video_with_comfyui(
    prompt: str,
    output_dir: str = OUTPUT_DIR
) -> List[str]:
    """
    Generates video frames using a ComfyUI workflow.

    Args:
        prompt: The optimized text prompt for video generation.
        output_dir: The directory to save the generated frames.

    Returns:
        A list of file paths to the generated frames.
    """
    logging.info("Starting video generation with ComfyUI.")
    if not os.path.exists("workflow_api.json"):
        logging.error("ComfyUI workflow file 'workflow_api.json' not found.")
        return []

    with open("workflow_api.json", "r") as f:
        workflow = json.load(f)

    # Find the positive prompt node
    clip_text_encode_nodes = {
        key: value
        for key, value in workflow.items()
        if value["class_type"] == "CLIPTextEncode"
    }
    if not clip_text_encode_nodes:
        logging.error("No 'CLIPTextEncode' node found in the workflow.")
        return []

    # Modify the prompt and set a random seed
    for key in clip_text_encode_nodes:
        workflow[key]["inputs"]["text"] = prompt

    # Set a new random seed for variation
    for key, value in workflow.items():
        if value["class_type"] in ["KSampler", "KSamplerAdvanced"]:
            workflow[key]["inputs"]["seed"] = int(time.time())

    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    try:
        ws.connect(f"ws://{COMFYUI_URL.split('//')[1]}/ws?clientId={client_id}")
    except Exception as e:
        logging.error(f"Failed to connect to ComfyUI WebSocket: {e}")
        return []

    def queue_prompt(prompt_workflow: Dict[str, Any], client_id: str) -> Optional[str]:
        try:
            response = requests.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": prompt_workflow, "client_id": client_id},
            )
            response.raise_for_status()
            return response.json()["prompt_id"]
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to queue prompt in ComfyUI: {e}")
            return None

    prompt_id = queue_prompt(workflow, client_id)
    if not prompt_id:
        ws.close()
        return []

    logging.info(f"Workflow queued with prompt ID: {prompt_id}")
    generated_files = []

    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break  # Execution is done
                elif message["type"] == "executed":
                    data = message["data"]
                    if "images" in data["output"]:
                        for img in data["output"]["images"]:
                            generated_files.append(
                                os.path.join(img["subfolder"], img["filename"])
                            )
            else:
                # Handle binary data if necessary (e.g., previews)
                continue
    finally:
        ws.close()

    logging.info(f"Generated {len(generated_files)} frames.")
    return generated_files


def compile_frames_to_video(
    frame_dir: str,
    output_file: str = "output.mp4",
    fps: int = 30
) -> bool:
    """
    Compiles a sequence of image frames into a video file using FFmpeg.

    Args:
        frame_dir: The directory containing the image frames.
        output_file: The path for the output video file.
        fps: The frames per second for the output video.

    Returns:
        True if the video was compiled successfully, False otherwise.
    """
    logging.info(f"Compiling frames from '{frame_dir}' into '{output_file}'.")
    frames = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]),
    )

    if not frames:
        logging.error("No frames found to compile.")
        return False

    # Using a temporary file for the frame list
    with open("framelist.txt", "w") as f:
        for frame in frames:
            f.write(f"file '{os.path.join(frame_dir, frame)}'\n")

    # FFmpeg command for ARM hardware
    command = [
        "ffmpeg",
        "-y",
        "-r",
        str(fps),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "framelist.txt",
        "-c:v",
        "libx264",  # A safe bet for ARM, hevc_videotoolbox is for Apple
        "-pix_fmt",
        "yuv420p",
        "-s",
        "1920x1080",
        output_file,
    ]

    try:
        subprocess.run(command, check=True)
        logging.info(f"Video successfully created: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {e}")
        return False
    finally:
        os.remove("framelist.txt")


def main(user_input: str, output_filename: str = "generated_video.mp4"):
    """
    The main workflow for generating a video from a user's text prompt.

    Args:
        user_input: The text prompt from the user.
        output_filename: The desired name for the final video file.
    """
    if not check_ffmpeg():
        return

    start_time = time.time()
    logging.info("--- Starting Video Generation Workflow ---")

    # 1. Optimize the prompt
    optimized_prompt = optimize_prompt_with_ollama(user_input)

    # 2. Generate frames
    frames_output_dir = os.path.join(OUTPUT_DIR, str(uuid.uuid4()))
    os.makedirs(frames_output_dir, exist_ok=True)
    generated_frames = generate_video_with_comfyui(optimized_prompt, frames_output_dir)

    if not generated_frames:
        logging.error("Frame generation failed. Aborting.")
        return

    # 3. Compile video
    final_video_path = os.path.join(OUTPUT_DIR, output_filename)
    success = compile_frames_to_video(frames_output_dir, final_video_path)

    if success:
        logging.info(f"--- Workflow Complete ---")
        logging.info(f"Final video available at: {final_video_path}")
    else:
        logging.error("--- Workflow Failed ---")

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a video from a text prompt.")
    parser.add_argument("prompt", type=str, help="The text prompt for the video.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="generated_video.mp4",
        help="The output filename for the video.",
    )
    args = parser.parse_args()

    main(args.prompt, args.output)
