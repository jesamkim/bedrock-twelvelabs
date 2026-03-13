"""
Experiment 2: Marengo Embed 3.0 - Multimodal Embedding
- Text embedding: direct
- Image embedding: extract frames from video -> embed
- Text-to-Image similarity search
- Pegasus + Marengo pipeline: video -> description -> text embedding

Bedrock Marengo Embed API format:
  Text:  {"inputType": "text", "text": {"inputText": "..."}}
  Image: {"inputType": "image", "image": {"mediaSource": {"base64String": "..."}}}
  Response: {"data": [{"embedding": [float, ...]}]}  dim=512
"""
import json
import base64
import subprocess
import os
import numpy as np
import boto3

REGION = "us-east-1"
MARENGO_ID = "us.twelvelabs.marengo-embed-3-0-v1:0"
PEGASUS_ID = "us.twelvelabs.pegasus-1-2-v1:0"
BUCKET = "bedrock-twelvelabs-poc-376278017302"
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
FRAMES_DIR = os.path.join(os.path.dirname(__file__), "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)


def embed_text(text: str) -> list[float]:
    body = {"inputType": "text", "text": {"inputText": text}}
    r = bedrock.invoke_model(
        modelId=MARENGO_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    result = json.loads(r["body"].read())
    return result["data"][0]["embedding"]


def embed_image_base64(img_b64: str) -> list[float]:
    body = {"inputType": "image", "image": {"mediaSource": {"base64String": img_b64}}}
    r = bedrock.invoke_model(
        modelId=MARENGO_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    result = json.loads(r["body"].read())
    return result["data"][0]["embedding"]


def embed_image_file(path: str) -> list[float]:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return embed_image_base64(b64)


def extract_frames(video_path: str, name: str, num_frames: int = 4) -> list[str]:
    """Extract evenly-spaced frames from video as JPEG files."""
    out_dir = os.path.join(FRAMES_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    # Get video duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    interval = duration / (num_frames + 1)

    paths = []
    for i in range(num_frames):
        ts = interval * (i + 1)
        out_path = os.path.join(out_dir, f"frame_{i:02d}.jpg")
        subprocess.run(
            ["ffmpeg", "-ss", str(ts), "-i", video_path,
             "-frames:v", "1", "-q:v", "2", "-y", out_path],
            capture_output=True,
        )
        if os.path.exists(out_path):
            paths.append(out_path)

    return paths


def invoke_pegasus(video_key: str, prompt: str) -> str:
    body = {
        "inputPrompt": prompt,
        "mediaSource": {
            "s3Location": {
                "uri": f"s3://{BUCKET}/{video_key}",
                "bucketOwner": ACCOUNT_ID,
            }
        },
    }
    r = bedrock.invoke_model(
        modelId=PEGASUS_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    return json.loads(r["body"].read()).get("message", "")


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def avg_embedding(embeddings: list[list[float]]) -> list[float]:
    return np.mean(embeddings, axis=0).tolist()


def main():
    videos = {
        "nature": "videos/nature.mp4",
        "city": "videos/city.mp4",
        "cooking": "videos/cooking.mp4",
    }

    print("=" * 70)
    print("Experiment 2: Marengo Embed 3.0 - Multimodal Embeddings")
    print(f"Model: {MARENGO_ID} | Region: {REGION}")
    print(f"Embedding dim: 512")
    print("=" * 70)

    # ── Step 1: Extract frames & generate image embeddings ──
    print("\n[Step 1] Frame extraction + Image embedding")
    video_frame_embeddings = {}
    for name, vkey in videos.items():
        vpath = os.path.join(os.path.dirname(__file__), vkey)
        print(f"\n  {name}: extracting 4 frames...")
        frames = extract_frames(vpath, name, num_frames=4)
        print(f"    frames: {len(frames)}")

        embeddings = []
        for fp in frames:
            try:
                emb = embed_image_file(fp)
                embeddings.append(emb)
                print(f"    {os.path.basename(fp)}: embedded (dim={len(emb)})")
            except Exception as e:
                print(f"    {os.path.basename(fp)}: ERROR - {e}")

        if embeddings:
            video_frame_embeddings[name] = {
                "frames": embeddings,
                "avg": avg_embedding(embeddings),
            }

    # ── Step 2: Text embeddings for search queries ──
    print(f"\n{'='*70}")
    print("[Step 2] Text-to-Image Similarity Search")
    print(f"{'='*70}")

    queries = [
        "woman watching hot air balloons at sunset",
        "two people talking by the river with laptops",
        "laboratory pipette experiment with green liquid",
        "cooking food in a kitchen",
        "beautiful nature landscape",
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        try:
            text_emb = embed_text(query)
            scores = {}
            for vname, data in video_frame_embeddings.items():
                scores[vname] = cosine_sim(text_emb, data["avg"])

            for vname, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                bar = "#" * int(score * 40)
                print(f"    {vname:10s}: {score:.4f} {bar}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # ── Step 3: Video-to-Video similarity (frame avg) ──
    print(f"\n{'='*70}")
    print("[Step 3] Video-to-Video Similarity (avg frame embedding)")
    print(f"{'='*70}")

    names = list(video_frame_embeddings.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i:]:
            sim = cosine_sim(
                video_frame_embeddings[n1]["avg"],
                video_frame_embeddings[n2]["avg"],
            )
            marker = " <<<" if n1 == n2 else ""
            print(f"  {n1:10s} <-> {n2:10s}: {sim:.4f}{marker}")

    # ── Step 4: Pegasus + Marengo pipeline ──
    print(f"\n{'='*70}")
    print("[Step 4] Pegasus -> Marengo Pipeline (Video Description Embedding)")
    print(f"{'='*70}")

    desc_embeddings = {}
    for name, vkey in videos.items():
        print(f"\n  {name}: Pegasus generating description...")
        try:
            desc = invoke_pegasus(vkey, "Describe this video in one detailed paragraph.")
            print(f"    Description: {desc[:150]}...")
            desc_emb = embed_text(desc)
            desc_embeddings[name] = desc_emb
            print(f"    Text embedding generated (dim={len(desc_emb)})")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Compare frame-based vs description-based embeddings
    if desc_embeddings:
        print(f"\n  --- Frame-avg vs Description-based similarity to queries ---")
        test_queries = [
            "hot air balloons in cappadocia",
            "people having conversation outdoors",
            "science experiment in lab",
        ]
        for query in test_queries:
            print(f"\n  Query: \"{query}\"")
            text_emb = embed_text(query)
            print(f"    {'Video':<12} {'Frame-avg':>10} {'Desc-based':>12}")
            for vname in names:
                frame_score = cosine_sim(text_emb, video_frame_embeddings[vname]["avg"])
                desc_score = cosine_sim(text_emb, desc_embeddings[vname]) if vname in desc_embeddings else 0
                print(f"    {vname:<12} {frame_score:>10.4f} {desc_score:>12.4f}")


if __name__ == "__main__":
    main()
