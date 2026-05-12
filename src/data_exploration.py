from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from config import DATASET_DIR, FIGURES_DIR, RESULTS_DIR, ensure_output_dirs


EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = ("NORMAL", "PNEUMONIA")
VALID_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}


def _list_images(directory: Path) -> list[Path]:
    return [file for file in directory.iterdir() if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS]


def check_dataset_structure() -> list[Path]:
    missing_paths = []
    for split in EXPECTED_SPLITS:
        for class_name in EXPECTED_CLASSES:
            class_dir = DATASET_DIR / split / class_name
            if not class_dir.exists():
                missing_paths.append(class_dir)
    return missing_paths


def collect_dataset_statistics() -> pd.DataFrame:
    rows = []
    for split in EXPECTED_SPLITS:
        for class_name in EXPECTED_CLASSES:
            class_dir = DATASET_DIR / split / class_name
            image_count = len(_list_images(class_dir))
            rows.append({"split": split, "class": class_name, "count": image_count})
    return pd.DataFrame(rows)


def save_class_distribution_plot(summary_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary_df, x="split", y="count", hue="class")
    plt.title("Class Distribution by Dataset Split")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close()


def save_example_grid(max_images_per_class: int = 4) -> None:
    sample_paths = []
    for class_name in EXPECTED_CLASSES:
        class_dir = DATASET_DIR / "train" / class_name
        class_images = _list_images(class_dir)[:max_images_per_class]
        sample_paths.extend((class_name, image_path) for image_path in class_images)

    if not sample_paths:
        return

    columns = max_images_per_class
    rows = len(EXPECTED_CLASSES)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    if rows == 1:
        axes = [axes]

    for row_idx, class_name in enumerate(EXPECTED_CLASSES):
        class_samples = [item for item in sample_paths if item[0] == class_name]
        for col_idx in range(columns):
            ax = axes[row_idx][col_idx] if columns > 1 else axes[row_idx]
            if col_idx < len(class_samples):
                image = Image.open(class_samples[col_idx][1]).convert("L")
                ax.imshow(image, cmap="gray")
                ax.set_title(class_name)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "example_images_grid.png", dpi=300)
    plt.close()


def main() -> None:
    ensure_output_dirs()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at {DATASET_DIR}. "
            "Please download the dataset and place it in data/chest_xray/."
        )

    missing_paths = check_dataset_structure()
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing expected dataset folders:\n{missing_text}")

    summary_df = collect_dataset_statistics()
    summary_df.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    print("\nDataset summary:")
    print(summary_df)
    print("\nPivot table (split x class):")
    print(summary_df.pivot(index="split", columns="class", values="count").fillna(0).astype(int))

    save_class_distribution_plot(summary_df)
    save_example_grid()

    print(f"\nSaved class distribution plot to: {FIGURES_DIR / 'class_distribution.png'}")
    print(f"Saved dataset summary CSV to: {RESULTS_DIR / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
