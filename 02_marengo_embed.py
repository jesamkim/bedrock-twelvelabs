"""
Experiment 2: Marengo Embed 3.0 - Multimodal Embedding

Bedrock Marengo Embed API:
  [Sync] InvokeModel - text, image (search query embedding)
    Text:  {"inputType": "text", "text": {"inputText": "..."}}
    Image: {"inputType": "image", "image": {"mediaSource": {"base64String": "..."}}}

  [Async] StartAsyncInvoke - video, audio, image, text (asset indexing)
    Video: {"inputType": "video", "video": {"mediaSource": {"s3Location": {...}}}}
    Response: {"data": [{"embedding": [...], "embeddingScope": "asset|clip",
               "embeddingOption": "visual|audio|transcription",
               "startSec": 0.0, "endSec": 19.35}]}
    Multi-vector: asset-level (3) + clip-level (N*3), all dim=512
"""
import json
import time
import numpy as np
import boto3

REGION = "us-east-1"
MARENGO_SYNC_ID = "us.twelvelabs.marengo-embed-3-0-v1:0"
MARENGO_ASYNC_ID = "twelvelabs.marengo-embed-3-0-v1:0"
PEGASUS_ID = "us.twelvelabs.pegasus-1-2-v1:0"
BUCKET = "bedrock-twelvelabs-poc-376278017302"
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def embed_text(text: str) -> list[float]:
    """Sync text embedding via InvokeModel."""
    body = {"inputType": "text", "text": {"inputText": text}}
    r = bedrock.invoke_model(
        modelId=MARENGO_SYNC_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    return json.loads(r["body"].read())["data"][0]["embedding"]


def embed_video_async(video_key: str, output_prefix: str) -> str:
    """Start async video embedding via StartAsyncInvoke. Returns invocation ARN."""
    body = {
        "inputType": "video",
        "video": {
            "mediaSource": {
                "s3Location": {
                    "uri": f"s3://{BUCKET}/{video_key}",
                    "bucketOwner": ACCOUNT_ID,
                }
            }
        },
    }
    r = bedrock.start_async_invoke(
        modelId=MARENGO_ASYNC_ID,
        modelInput=body,
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": f"s3://{BUCKET}/{output_prefix}"
            }
        },
    )
    return r["invocationArn"]


def wait_async_invoke(arn: str, poll_interval: int = 3) -> dict:
    """Poll until async invoke completes, return result metadata."""
    while True:
        r = bedrock.get_async_invoke(invocationArn=arn)
        if r["status"] in ("Completed", "Failed"):
            return r
        time.sleep(poll_interval)


def load_async_output(s3_uri: str) -> dict:
    """Download output.json from S3."""
    s3 = boto3.client("s3", region_name=REGION)
    # Parse s3://bucket/prefix/invocationId -> bucket, key
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, prefix = parts[0], parts[1]
    key = f"{prefix}/output.json" if not prefix.endswith("/output.json") else prefix

    r = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(r["Body"].read())


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


def parse_video_embeddings(data: dict) -> dict:
    """Parse async output into structured embeddings."""
    segments = data["data"]
    result = {"asset": {}, "clips": []}

    clip_map = {}
    for seg in segments:
        scope = seg["embeddingScope"]
        option = seg["embeddingOption"]
        emb = seg["embedding"]
        start = seg["startSec"]
        end = seg["endSec"]

        if scope == "asset":
            result["asset"][option] = emb
        else:
            clip_key = (start, end)
            if clip_key not in clip_map:
                clip_map[clip_key] = {"startSec": start, "endSec": end}
            clip_map[clip_key][option] = emb

    result["clips"] = sorted(clip_map.values(), key=lambda x: x["startSec"])
    return result


def main():
    videos = {
        "nature": "videos/nature.mp4",
        "city": "videos/city.mp4",
        "cooking": "videos/cooking.mp4",
    }

    print("=" * 70)
    print("Experiment 2: Marengo Embed 3.0 - Video Embedding (Async)")
    print(f"Sync Model:  {MARENGO_SYNC_ID} (text/image query)")
    print(f"Async Model: {MARENGO_ASYNC_ID} (video indexing)")
    print(f"Region: {REGION} | Embedding dim: 512")
    print("=" * 70)

    # ── Step 1: Async video embedding ──
    print("\n[Step 1] Async Video Embedding (StartAsyncInvoke)")

    invocations = {}
    for name, vkey in videos.items():
        print(f"  {name}: starting async invoke...")
        arn = embed_video_async(vkey, f"async-embed/{name}/")
        invocations[name] = arn
        print(f"    ARN: {arn}")

    print("\n  Waiting for completion...")
    video_embeddings = {}
    for name, arn in invocations.items():
        result = wait_async_invoke(arn)
        s3_uri = result["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        print(f"  {name}: {result['status']} -> {s3_uri}")

        if result["status"] == "Completed":
            output = load_async_output(s3_uri)
            parsed = parse_video_embeddings(output)
            video_embeddings[name] = parsed

            n_clips = len(parsed["clips"])
            modalities = list(parsed["asset"].keys())
            print(f"    asset modalities: {modalities}")
            print(f"    clips: {n_clips} segments")
            for clip in parsed["clips"]:
                print(f"      [{clip['startSec']:6.2f}s - {clip['endSec']:6.2f}s]")

    # ── Step 2: Text-to-Video search (asset-level visual embedding) ──
    print(f"\n{'='*70}")
    print("[Step 2] Text-to-Video Search (Asset-level Visual Embedding)")
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
        text_emb = embed_text(query)
        scores = {}
        for vname, data in video_embeddings.items():
            scores[vname] = cosine_sim(text_emb, data["asset"]["visual"])

        for vname, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(max(score, 0) * 40)
            print(f"    {vname:10s}: {score:.4f} {bar}")

    # ── Step 3: Multi-modal asset similarity ──
    print(f"\n{'='*70}")
    print("[Step 3] Multi-modal Asset Similarity")
    print(f"{'='*70}")

    for modality in ["visual", "audio", "transcription"]:
        print(f"\n  --- {modality} ---")
        names = list(video_embeddings.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                e1 = video_embeddings[n1]["asset"].get(modality)
                e2 = video_embeddings[n2]["asset"].get(modality)
                if e1 and e2:
                    sim = cosine_sim(e1, e2)
                    print(f"    {n1:10s} <-> {n2:10s}: {sim:.4f}")

    # ── Step 4: Clip-level temporal search ──
    print(f"\n{'='*70}")
    print("[Step 4] Clip-level Temporal Search")
    print(f"{'='*70}")

    temporal_queries = [
        "hot air balloons floating in the sky",
        "woman standing on a hill",
        "people gesturing with hands while talking",
    ]

    for query in temporal_queries:
        print(f"\n  Query: \"{query}\"")
        text_emb = embed_text(query)

        all_clips = []
        for vname, data in video_embeddings.items():
            for clip in data["clips"]:
                visual_emb = clip.get("visual")
                if visual_emb:
                    score = cosine_sim(text_emb, visual_emb)
                    all_clips.append((vname, clip["startSec"], clip["endSec"], score))

        all_clips.sort(key=lambda x: x[3], reverse=True)
        for vname, start, end, score in all_clips[:5]:
            bar = "#" * int(max(score, 0) * 40)
            print(f"    {vname:10s} [{start:5.1f}s-{end:5.1f}s]: {score:.4f} {bar}")

    # ── Step 5: Compare 3 approaches ──
    print(f"\n{'='*70}")
    print("[Step 5] 3-Way Comparison: Async Video vs Pegasus+Marengo vs Text Query")
    print(f"{'='*70}")

    desc_embeddings = {}
    for name, vkey in videos.items():
        print(f"\n  {name}: Pegasus generating description...")
        desc = invoke_pegasus(vkey, "Describe this video in one detailed paragraph.")
        print(f"    Description: {desc[:120]}...")
        desc_embeddings[name] = embed_text(desc)

    comparison_queries = [
        "hot air balloons in cappadocia",
        "people having conversation outdoors",
        "science experiment in lab",
    ]

    print(f"\n  {'Query':<40} {'Video':<10} {'AsyncVid':>10} {'PegDesc':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
    for query in comparison_queries:
        text_emb = embed_text(query)
        for vname in video_embeddings:
            async_score = cosine_sim(text_emb, video_embeddings[vname]["asset"]["visual"])
            desc_score = cosine_sim(text_emb, desc_embeddings[vname])
            print(f"  {query:<40} {vname:<10} {async_score:>10.4f} {desc_score:>10.4f}")
        print()


if __name__ == "__main__":
    main()
