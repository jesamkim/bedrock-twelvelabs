"""Generate result charts for README."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["figure.dpi"] = 150

OUT_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "nature": "#2ecc71",
    "city": "#3498db",
    "cooking": "#e74c3c",
}


def chart_text_to_image_search():
    """Chart 1: Text-to-Image similarity (frame-avg)."""
    queries = [
        "woman watching\nhot air balloons",
        "two people talking\nby the river",
        "laboratory pipette\nexperiment",
        "cooking food\nin a kitchen",
        "beautiful nature\nlandscape",
    ]
    data = {
        "nature":  [0.0633, -0.0124, -0.0515, -0.0411, 0.0185],
        "city":    [-0.0092, 0.1278, -0.0537, -0.0522, 0.0200],
        "cooking": [-0.0650, -0.0350, 0.0673, 0.0047, -0.0145],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(queries))
    width = 0.25

    for i, (name, scores) in enumerate(data.items()):
        bars = ax.bar(x + i * width, scores, width, label=name, color=COLORS[name], alpha=0.85)
        for bar, score in zip(bars, scores):
            if abs(score) > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{score:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Search Query")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Marengo Embed 3.0: Text-to-Image Search (Frame-avg)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(queries, fontsize=8)
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylim(-0.1, 0.18)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_frame_search.png"))
    plt.close()
    print("chart_frame_search.png saved")


def chart_pipeline_comparison():
    """Chart 2: Frame-avg vs Pegasus+Marengo pipeline comparison."""
    queries = [
        "hot air balloons\nin cappadocia",
        "people having\nconversation outdoors",
        "science experiment\nin lab",
    ]

    frame_avg = {
        "nature":  [0.0428, 0.0056, -0.0324],
        "city":    [-0.0510, 0.0687, -0.0431],
        "cooking": [-0.0584, -0.0275, 0.0677],
    }
    desc_based = {
        "nature":  [0.4207, 0.2903, 0.2219],
        "city":    [0.1762, 0.6164, 0.2543],
        "cooking": [0.1166, 0.1576, 0.5622],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, query in enumerate(queries):
        ax = axes[idx]
        videos = list(frame_avg.keys())
        x = np.arange(len(videos))
        width = 0.35

        frame_vals = [frame_avg[v][idx] for v in videos]
        desc_vals = [desc_based[v][idx] for v in videos]

        bars1 = ax.bar(x - width / 2, frame_vals, width, label="Frame-avg",
                       color="#95a5a6", alpha=0.8)
        bars2 = ax.bar(x + width / 2, desc_vals, width, label="Pegasus+Marengo",
                       color="#e67e22", alpha=0.9)

        for bar, val in zip(bars1, frame_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars2, desc_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_title(query, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(videos, fontsize=9)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    axes[0].set_ylabel("Cosine Similarity")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Frame-avg vs Pegasus+Marengo Pipeline: Search Accuracy Comparison",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_pipeline_comparison.png"))
    plt.close()
    print("chart_pipeline_comparison.png saved")


def chart_video_similarity_matrix():
    """Chart 3: Video-to-Video similarity heatmap."""
    labels = ["nature", "city", "cooking"]
    matrix = np.array([
        [1.0000, 0.7708, 0.7937],
        [0.7708, 1.0000, 0.7456],
        [0.7937, 0.7456, 1.0000],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.7, vmax=1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if matrix[i, j] > 0.9 else "black"
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_title("Video-to-Video Similarity (Frame-avg Embedding)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_similarity_matrix.png"))
    plt.close()
    print("chart_similarity_matrix.png saved")


def chart_pegasus_features():
    """Chart 4: Pegasus capability summary."""
    features = ["English\nSummary", "Korean\nSummary", "Object\nDetection", "Timestamp\nExtraction"]
    # 1 = success, 0 = fail
    results = {
        "nature":  [1, 1, 1, 1],
        "city":    [1, 1, 1, 1],
        "cooking": [1, 1, 1, 1],
    }

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(features))
    width = 0.25

    for i, (name, vals) in enumerate(results.items()):
        ax.bar(x + i * width, vals, width, label=name, color=COLORS[name], alpha=0.85)

    ax.set_ylabel("Success (1=Pass)")
    ax.set_title("Pegasus v1.2: Video Understanding Capabilities")
    ax.set_xticks(x + width)
    ax.set_xticklabels(features)
    ax.set_ylim(0, 1.3)
    ax.legend()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_pegasus_features.png"))
    plt.close()
    print("chart_pegasus_features.png saved")


if __name__ == "__main__":
    chart_text_to_image_search()
    chart_pipeline_comparison()
    chart_video_similarity_matrix()
    chart_pegasus_features()
    print("\nAll charts generated in assets/")
