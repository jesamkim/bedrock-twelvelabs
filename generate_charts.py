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


def chart_async_video_search():
    """Chart 1: Async video embedding - text-to-video search."""
    queries = [
        "woman watching\nhot air balloons",
        "two people talking\nby the river",
        "laboratory pipette\nexperiment",
        "cooking food\nin a kitchen",
        "beautiful nature\nlandscape",
    ]
    data = {
        "nature":  [0.1404, -0.0218, -0.0828, -0.0663, 0.0491],
        "city":    [0.0046, 0.1311, -0.0662, -0.0667, 0.0442],
        "cooking": [-0.0695, -0.0751, 0.1497, -0.0154, -0.0221],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(queries))
    width = 0.25

    for i, (name, scores) in enumerate(data.items()):
        bars = ax.bar(x + i * width, scores, width, label=name, color=COLORS[name], alpha=0.85)
        for bar, score in zip(bars, scores):
            if abs(score) > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{score:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Search Query")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Marengo Embed 3.0: Text-to-Video Search (Async Video Embedding)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(queries, fontsize=8)
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylim(-0.12, 0.22)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_async_video_search.png"))
    plt.close()
    print("chart_async_video_search.png saved")


def chart_3way_comparison():
    """Chart 2: Async video vs Pegasus+Marengo comparison."""
    queries = [
        "hot air balloons\nin cappadocia",
        "people having\nconversation outdoors",
        "science experiment\nin lab",
    ]

    async_vid = {
        "nature":  [0.1347, 0.0024, -0.0572],
        "city":    [-0.0541, 0.0671, -0.0493],
        "cooking": [-0.0819, -0.0272, 0.0993],
    }
    peg_desc = {
        "nature":  [0.4560, 0.3421, 0.2047],
        "city":    [0.1955, 0.6361, 0.2491],
        "cooking": [0.1380, 0.1457, 0.5337],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, query in enumerate(queries):
        ax = axes[idx]
        videos = list(async_vid.keys())
        x = np.arange(len(videos))
        width = 0.35

        av_vals = [async_vid[v][idx] for v in videos]
        pd_vals = [peg_desc[v][idx] for v in videos]

        bars1 = ax.bar(x - width / 2, av_vals, width, label="Async Video Embed",
                       color="#3498db", alpha=0.8)
        bars2 = ax.bar(x + width / 2, pd_vals, width, label="Pegasus+Marengo",
                       color="#e67e22", alpha=0.9)

        for bar, val in zip(bars1, av_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars2, pd_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_title(query, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(videos, fontsize=9)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    axes[0].set_ylabel("Cosine Similarity")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Async Video Embed vs Pegasus+Marengo Pipeline: Search Accuracy",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_3way_comparison.png"))
    plt.close()
    print("chart_3way_comparison.png saved")


def chart_video_similarity_matrix():
    """Chart 3: Video-to-Video similarity heatmap (async visual embedding)."""
    labels = ["nature", "city", "cooking"]
    matrix = np.array([
        [1.0000, 0.6262, 0.5957],
        [0.6262, 1.0000, 0.6063],
        [0.5957, 0.6063, 1.0000],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if matrix[i, j] > 0.85 else "black"
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_title("Video-to-Video Similarity (Async Visual Embedding)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_similarity_matrix.png"))
    plt.close()
    print("chart_similarity_matrix.png saved")


def chart_clip_temporal_search():
    """Chart 4: Clip-level temporal search results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    clip_data = [
        {
            "query": '"hot air balloons\nfloating in the sky"',
            "clips": [
                ("nature\n6.5-13.0s", 0.1023),
                ("nature\n0.0-6.5s", 0.1012),
                ("nature\n13.0-19.2s", 0.1002),
                ("city\n13.0-19.2s", -0.0236),
                ("cooking\n0.0-7.0s", -0.0419),
            ],
        },
        {
            "query": '"woman standing\non a hill"',
            "clips": [
                ("nature\n0.0-6.5s", 0.0863),
                ("nature\n6.5-13.0s", 0.0854),
                ("nature\n13.0-19.2s", 0.0812),
                ("city\n13.0-19.2s", 0.0111),
                ("cooking\n0.0-7.0s", -0.0112),
            ],
        },
        {
            "query": '"people gesturing\nwith hands"',
            "clips": [
                ("city\n13.0-19.2s", 0.0604),
                ("city\n19.2-25.5s", 0.0419),
                ("city\n6.5-13.0s", 0.0240),
                ("nature\n0.0-6.5s", -0.0072),
                ("cooking\n0.0-7.0s", -0.0293),
            ],
        },
    ]

    for idx, d in enumerate(clip_data):
        ax = axes[idx]
        labels = [c[0] for c in d["clips"]]
        scores = [c[1] for c in d["clips"]]

        colors_list = []
        for label in labels:
            if label.startswith("nature"):
                colors_list.append(COLORS["nature"])
            elif label.startswith("city"):
                colors_list.append(COLORS["city"])
            else:
                colors_list.append(COLORS["cooking"])

        bars = ax.barh(range(len(labels)), scores, color=colors_list, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(d["query"], fontsize=9)
        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.invert_yaxis()

        for bar, score in zip(bars, scores):
            ax.text(max(score, 0) + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va="center", fontsize=7)

    fig.suptitle("Clip-level Temporal Search (Top 5 clips per query)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_clip_temporal.png"))
    plt.close()
    print("chart_clip_temporal.png saved")


def chart_pegasus_features():
    """Chart 5: Pegasus capability summary."""
    features = ["English\nSummary", "Korean\nSummary", "Object\nDetection", "Timestamp\nExtraction"]
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


def chart_api_comparison():
    """Chart 6: API method comparison - sync vs async embedding scores."""
    # Correct match scores only (query matches target video)
    methods = ["Async Video\n(visual)", "Pegasus+\nMarengo"]
    scores_per_video = {
        "nature\n(balloons)":  [0.1347, 0.4560],
        "city\n(conversation)":    [0.0671, 0.6361],
        "cooking\n(lab)": [0.0993, 0.5337],
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(scores_per_video))
    width = 0.3

    colors_methods = ["#3498db", "#e67e22"]
    for i, method in enumerate(methods):
        vals = [v[i] for v in scores_per_video.values()]
        bars = ax.bar(x + i * width, vals, width, label=method, color=colors_methods[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Cosine Similarity (correct match)")
    ax.set_title("Correct Match Score: Async Video Embed vs Pegasus+Marengo")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(scores_per_video.keys())
    ax.legend()
    ax.set_ylim(0, 0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_api_comparison.png"))
    plt.close()
    print("chart_api_comparison.png saved")


if __name__ == "__main__":
    chart_async_video_search()
    chart_3way_comparison()
    chart_video_similarity_matrix()
    chart_clip_temporal_search()
    chart_pegasus_features()
    chart_api_comparison()
    print("\nAll charts generated in assets/")
