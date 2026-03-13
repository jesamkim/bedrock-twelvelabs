"""
Experiment 1: Pegasus v1.2 - Video Q&A and Summarization
Bedrock InvokeModel with S3 video input
"""
import json
import boto3

REGION = "us-east-1"
MODEL_ID = "us.twelvelabs.pegasus-1-2-v1:0"
BUCKET = "bedrock-twelvelabs-poc-376278017302"
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def invoke_pegasus(video_key: str, prompt: str) -> dict:
    body = {
        "inputPrompt": prompt,
        "mediaSource": {
            "s3Location": {
                "uri": f"s3://{BUCKET}/{video_key}",
                "bucketOwner": ACCOUNT_ID,
            }
        },
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    return json.loads(response["body"].read())


def main():
    videos = {
        "nature": "videos/nature.mp4",
        "city": "videos/city.mp4",
        "cooking": "videos/cooking.mp4",
    }

    prompts = [
        ("summary_en", "Describe this video in detail. What is happening?"),
        ("summary_ko", "이 비디오의 내용을 한국어로 자세히 설명해주세요."),
        ("objects", "What are the main objects or subjects visible in this video?"),
        ("timestamps", "List the key moments in this video with approximate timestamps."),
    ]

    print("=" * 70)
    print("Experiment 1: Pegasus v1.2 - Video Understanding")
    print(f"Model: {MODEL_ID} | Region: {REGION}")
    print("=" * 70)

    for video_name, video_key in videos.items():
        print(f"\n{'='*70}")
        print(f"Video: {video_name} ({video_key})")
        print(f"{'='*70}")

        for prompt_type, prompt_text in prompts:
            print(f"\n--- [{prompt_type}] {prompt_text[:60]} ---")
            try:
                result = invoke_pegasus(video_key, prompt_text)
                text = result.get("message", "")
                stop = result.get("stopReason", "?")
                print(f"[stopReason: {stop}]")
                print(text[:800])
            except Exception as e:
                print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
