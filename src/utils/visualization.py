from pathlib import Path
import matplotlib.pyplot as plt

def _highlight_best(ax, x_values, y_values, mode: str = "max", label: str = "best"):
    if not y_values:
        return
    if mode == "max":
        best_idx = max(range(len(y_values)), key=lambda i: y_values[i])
    else:
        best_idx = min(range(len(y_values)), key=lambda i: y_values[i])

    best_x = x_values[best_idx]
    best_y = y_values[best_idx]

    ax.scatter([best_x],[best_y],s=60,marker="o",zorder=5,label=f"{label}: epoch {best_x}")
    ax.annotate(f"epoch {best_x}\n{best_y:.4f}",xy=(best_x, best_y),xytext=(8, 8),textcoords="offset points",fontsize=9)

def plot_loss_history(history: dict, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], linewidth=2, marker="o", markersize=4, label="Train Loss")
    ax.plot(epochs, history["val_loss"], linewidth=2, marker="o", markersize=4, label="Val Loss")

    _highlight_best(ax, epochs, history["val_loss"], mode="min", label="best val loss")

    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_accuracy_history(history: dict, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_acc"]) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], linewidth=2, marker="o", markersize=4, label="Train Accuracy")
    ax.plot(epochs, history["val_acc"], linewidth=2, marker="o", markersize=4, label="Val Accuracy")

    _highlight_best(ax, epochs, history["val_acc"], mode="max", label="best val acc")

    ax.set_title("Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_loss_history(history, output_path.parent / (output_path.stem + "_loss.png"))
    plot_accuracy_history(history, output_path.parent / (output_path.stem + "_acc.png"))