import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from config import EPOCHS, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, ensure_output_dirs
from model_baseline import build_baseline_model
from preprocessing import create_datasets


def save_training_plots(history) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_loss.png", dpi=300)
    plt.close()


def save_confusion_matrix_plot(model, test_ds, class_names) -> None:
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)
        y_true.extend(labels.numpy().flatten().astype(int))
        y_pred.extend(predictions)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def save_test_metrics(test_results) -> None:
    metric_names = ["loss", "accuracy", "precision", "recall"]
    metrics = dict(zip(metric_names, test_results))

    with open(RESULTS_DIR / "baseline_metrics.txt", "w", encoding="utf-8") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value:.4f}\n")


def main() -> None:
    ensure_output_dirs()

    train_ds, val_ds, test_ds, class_names = create_datasets()
    model = build_baseline_model()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1,
    )

    model.save(MODELS_DIR / "baseline_cnn.keras")
    save_training_plots(history)

    test_results = model.evaluate(test_ds, verbose=1)
    save_test_metrics(test_results)
    save_confusion_matrix_plot(model, test_ds, class_names)

    print(f"Saved model to: {MODELS_DIR / 'baseline_cnn.keras'}")
    print(f"Saved history CSV to: {RESULTS_DIR / 'training_history.csv'}")
    print(f"Saved metrics to: {RESULTS_DIR / 'baseline_metrics.txt'}")
    print(f"Saved confusion matrix to: {FIGURES_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
