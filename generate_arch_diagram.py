"""Generate architecture diagram as SVG then convert to PNG."""
import cairosvg
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT_DIR, exist_ok=True)

SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 520" width="900" height="520">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0f1923"/>
      <stop offset="100%" stop-color="#1a2a3a"/>
    </linearGradient>
    <linearGradient id="boxS3" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#3F8624"/>
      <stop offset="100%" stop-color="#6AAF35"/>
    </linearGradient>
    <linearGradient id="boxPeg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#C7511F"/>
      <stop offset="100%" stop-color="#E8772E"/>
    </linearGradient>
    <linearGradient id="boxMar" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1B66C9"/>
      <stop offset="100%" stop-color="#4A90D9"/>
    </linearGradient>
    <linearGradient id="boxVec" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#8E44AD"/>
      <stop offset="100%" stop-color="#BB6BD9"/>
    </linearGradient>
    <filter id="shadow">
      <feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.3"/>
    </filter>
  </defs>

  <rect width="900" height="520" fill="url(#bg)" rx="12"/>

  <!-- Title -->
  <text x="450" y="40" text-anchor="middle" fill="#ffffff" font-size="18" font-weight="bold" font-family="Arial, sans-serif">Bedrock Twelve Labs PoC Architecture (us-east-1)</text>

  <!-- Region box -->
  <rect x="30" y="55" width="840" height="450" rx="8" fill="none" stroke="#4a6a8a" stroke-width="1.5" stroke-dasharray="6,4"/>
  <text x="50" y="75" fill="#7a9aba" font-size="12" font-family="Arial, sans-serif">AWS Region: us-east-1</text>

  <!-- S3 Bucket -->
  <rect x="60" y="100" width="160" height="160" rx="10" fill="url(#boxS3)" filter="url(#shadow)"/>
  <text x="140" y="130" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="bold" font-family="Arial, sans-serif">Amazon S3</text>
  <text x="140" y="155" text-anchor="middle" fill="#e0e0e0" font-size="10" font-family="Arial, sans-serif">bedrock-twelvelabs-poc-*</text>
  <text x="140" y="180" text-anchor="middle" fill="#c0d0c0" font-size="11" font-family="Arial, sans-serif">nature.mp4 (19s)</text>
  <text x="140" y="200" text-anchor="middle" fill="#c0d0c0" font-size="11" font-family="Arial, sans-serif">city.mp4 (38s)</text>
  <text x="140" y="220" text-anchor="middle" fill="#c0d0c0" font-size="11" font-family="Arial, sans-serif">cooking.mp4 (14s)</text>

  <!-- Pegasus -->
  <rect x="340" y="90" width="220" height="110" rx="10" fill="url(#boxPeg)" filter="url(#shadow)"/>
  <text x="450" y="120" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="bold" font-family="Arial, sans-serif">Pegasus v1.2</text>
  <text x="450" y="142" text-anchor="middle" fill="#ffe0c0" font-size="11" font-family="Arial, sans-serif">Video-to-Text Generation</text>
  <text x="450" y="162" text-anchor="middle" fill="#e0d0c0" font-size="10" font-family="Arial, sans-serif">Summary / Q&amp;A / Timestamps</text>
  <text x="450" y="182" text-anchor="middle" fill="#d0c0b0" font-size="9" font-family="monospace">us.twelvelabs.pegasus-1-2-v1:0</text>

  <!-- Marengo -->
  <rect x="340" y="230" width="220" height="110" rx="10" fill="url(#boxMar)" filter="url(#shadow)"/>
  <text x="450" y="260" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="bold" font-family="Arial, sans-serif">Marengo Embed 3.0</text>
  <text x="450" y="282" text-anchor="middle" fill="#c0d8ff" font-size="11" font-family="Arial, sans-serif">Text + Image Embedding</text>
  <text x="450" y="302" text-anchor="middle" fill="#b0c8e0" font-size="10" font-family="Arial, sans-serif">dim=512 / Cosine Similarity</text>
  <text x="450" y="322" text-anchor="middle" fill="#a0b8d0" font-size="9" font-family="monospace">us.twelvelabs.marengo-embed-3-0-v1:0</text>

  <!-- Vector DB -->
  <rect x="670" y="230" width="170" height="110" rx="10" fill="url(#boxVec)" filter="url(#shadow)"/>
  <text x="755" y="265" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="bold" font-family="Arial, sans-serif">Vector DB</text>
  <text x="755" y="287" text-anchor="middle" fill="#e0d0f0" font-size="11" font-family="Arial, sans-serif">OpenSearch /</text>
  <text x="755" y="305" text-anchor="middle" fill="#e0d0f0" font-size="11" font-family="Arial, sans-serif">Pinecone</text>
  <text x="755" y="325" text-anchor="middle" fill="#d0c0e0" font-size="10" font-family="Arial, sans-serif">Similarity Search</text>

  <!-- Output -->
  <rect x="670" y="90" width="170" height="110" rx="10" fill="none" stroke="#e8772e" stroke-width="2"/>
  <text x="755" y="120" text-anchor="middle" fill="#e8772e" font-size="13" font-weight="bold" font-family="Arial, sans-serif">Output</text>
  <text x="755" y="145" text-anchor="middle" fill="#c0c0c0" font-size="10" font-family="Arial, sans-serif">English/Korean Summary</text>
  <text x="755" y="163" text-anchor="middle" fill="#c0c0c0" font-size="10" font-family="Arial, sans-serif">Timestamped Moments</text>
  <text x="755" y="181" text-anchor="middle" fill="#c0c0c0" font-size="10" font-family="Arial, sans-serif">Object Detection</text>

  <!-- Arrows: S3 -> Pegasus -->
  <line x1="220" y1="160" x2="335" y2="140" stroke="#6AAF35" stroke-width="2.5" marker-end="url(#arrowG)"/>
  <!-- S3 -> Marengo (via frame extraction) -->
  <line x1="220" y1="220" x2="335" y2="270" stroke="#6AAF35" stroke-width="2.5" marker-end="url(#arrowG)"/>
  <!-- Pegasus -> Marengo (description text) -->
  <line x1="450" y1="200" x2="450" y2="225" stroke="#E8772E" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arrowO)"/>
  <!-- Pegasus -> Output -->
  <line x1="560" y1="145" x2="665" y2="145" stroke="#E8772E" stroke-width="2.5" marker-end="url(#arrowO)"/>
  <!-- Marengo -> Vector DB -->
  <line x1="560" y1="285" x2="665" y2="285" stroke="#4A90D9" stroke-width="2.5" marker-end="url(#arrowB)"/>

  <!-- Arrow markers -->
  <defs>
    <marker id="arrowG" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#6AAF35"/>
    </marker>
    <marker id="arrowO" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#E8772E"/>
    </marker>
    <marker id="arrowB" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#4A90D9"/>
    </marker>
  </defs>

  <!-- Labels on arrows -->
  <text x="270" y="140" fill="#a0d080" font-size="9" font-family="Arial, sans-serif" transform="rotate(-8, 270, 140)">video (S3 URI)</text>
  <text x="255" y="260" fill="#a0d080" font-size="9" font-family="Arial, sans-serif" transform="rotate(15, 255, 260)">frames (JPEG)</text>
  <text x="458" y="216" fill="#e8a060" font-size="9" font-family="Arial, sans-serif">description text</text>
  <text x="600" y="138" fill="#e8a060" font-size="9" font-family="Arial, sans-serif">text response</text>
  <text x="600" y="278" fill="#80b0e0" font-size="9" font-family="Arial, sans-serif">embedding</text>

  <!-- Pipeline labels -->
  <rect x="60" y="380" width="400" height="100" rx="8" fill="#1e2e3e" stroke="#3a5a7a" stroke-width="1"/>
  <text x="80" y="405" fill="#e8772e" font-size="12" font-weight="bold" font-family="Arial, sans-serif">Pipeline A: Video Understanding (Pegasus)</text>
  <text x="80" y="425" fill="#b0b0b0" font-size="10" font-family="Arial, sans-serif">S3 Video -> Pegasus -> Summary / Q&amp;A / Timestamps</text>
  <text x="80" y="450" fill="#4A90D9" font-size="12" font-weight="bold" font-family="Arial, sans-serif">Pipeline B: Video Search (Pegasus + Marengo)</text>
  <text x="80" y="470" fill="#b0b0b0" font-size="10" font-family="Arial, sans-serif">S3 Video -> Pegasus desc -> Marengo embed -> Vector DB</text>

  <!-- Bedrock badge -->
  <rect x="580" y="380" width="270" height="100" rx="8" fill="#1e2e3e" stroke="#3a5a7a" stroke-width="1"/>
  <text x="715" y="405" text-anchor="middle" fill="#FF9900" font-size="13" font-weight="bold" font-family="Arial, sans-serif">Amazon Bedrock</text>
  <text x="715" y="425" text-anchor="middle" fill="#a0a0a0" font-size="10" font-family="Arial, sans-serif">InvokeModel API</text>
  <text x="715" y="445" text-anchor="middle" fill="#a0a0a0" font-size="10" font-family="Arial, sans-serif">IAM Authentication</text>
  <text x="715" y="465" text-anchor="middle" fill="#a0a0a0" font-size="10" font-family="Arial, sans-serif">us-east-1 (Marengo) / global (Pegasus)</text>
</svg>
"""

svg_path = os.path.join(OUT_DIR, "architecture.svg")
png_path = os.path.join(OUT_DIR, "architecture.png")

with open(svg_path, "w") as f:
    f.write(SVG)

cairosvg.svg2png(url=svg_path, write_to=png_path, scale=2)
print(f"Architecture diagram saved: {png_path}")
