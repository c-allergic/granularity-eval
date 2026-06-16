"""
Generate captions for all (image, cell) combinations using two models:
  - InternVL2-8B (local, L40)
  - GPT-4o (API)

Output: results/captions/{model}/{image_id}/{cell_id}.txt
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
PROMPT_FILE = PROJECT_ROOT / "prompts" / "prompts.json"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "captions"

MAX_TOKENS_MAP = {"G1": 64, "G2": 128, "G3": 256}


def load_prompts():
    with open(PROMPT_FILE) as f:
        return json.load(f)


def find_images():
    """Find all images in data/images/, supporting nested category subdirs."""
    images = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for p in IMAGE_DIR.rglob(ext):
            # Use relative path from IMAGE_DIR as image_id
            rel = p.relative_to(IMAGE_DIR)
            image_id = str(rel.with_suffix("")).replace("/", "_")
            category = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            images[image_id] = {"path": str(p), "category": category}
    return images


def generate_internvl(images, prompts, model_name="OpenGVLab/InternVL2-8B"):
    """Generate captions using InternVL2-8B."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {model_name}...")
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_out_dir = OUTPUT_DIR / "internvl2-8b"

    for image_id, info in tqdm(list(images.items()), desc="InternVL2-8B"):
        img = Image.open(info["path"]).convert("RGB")

        for cell_id, cell in prompts.items():
            out_path = model_out_dir / image_id / f"{cell_id}.txt"
            if out_path.exists():
                continue  # skip already generated

            grain = cell["granularity"][:2].upper()  # G1, G2, G3
            max_tokens = MAX_TOKENS_MAP.get(grain, 128)
            system_prompt = cell["prompt"]

            # InternVL2 chat format
            response = model.chat(
                tokenizer, img, system_prompt,
                generation_config={"max_new_tokens": max_tokens, "temperature": 0.7}
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(response.strip())

    print(f"InternVL2-8B captions saved to {model_out_dir}")


def generate_gpt4o(images, prompts):
    """Generate captions using GPT-4o API."""
    from openai import OpenAI

    client = OpenAI()
    model_out_dir = OUTPUT_DIR / "gpt4o"

    for image_id, info in tqdm(list(images.items()), desc="GPT-4o"):
        # Encode image as base64 for API
        import base64
        with open(info["path"], "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        for cell_id, cell in prompts.items():
            out_path = model_out_dir / image_id / f"{cell_id}.txt"
            if out_path.exists():
                continue

            grain = cell["granularity"][:2].upper()
            max_tokens = MAX_TOKENS_MAP.get(grain, 128)

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": cell["prompt"]},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    }],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                text = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"GPT-4o error {image_id}/{cell_id}: {e}")
                time.sleep(5)
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(text)

    print(f"GPT-4o captions saved to {model_out_dir}")


def main():
    if not IMAGE_DIR.exists():
        print(f"Image directory not found: {IMAGE_DIR}")
        print("Place images in data/images/{category}/ (natural, doc_poster, art)")
        sys.exit(1)

    prompts = load_prompts()
    images = find_images()
    print(f"Found {len(images)} images across {len(set(i['category'] for i in images.values()))} categories")
    print(f"Generating {len(images)} x {len(prompts)} = {len(images) * len(prompts)} captions per model")

    if "--internvl" in sys.argv:
        generate_internvl(images, prompts)
    elif "--gpt4o" in sys.argv:
        generate_gpt4o(images, prompts)
    else:
        # Default: run both
        generate_internvl(images, prompts)
        generate_gpt4o(images, prompts)


if __name__ == "__main__":
    main()
