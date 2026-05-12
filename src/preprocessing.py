import tensorflow as tf

from config import BATCH_SIZE, DATASET_DIR, IMAGE_SIZE, RANDOM_SEED


AUTOTUNE = tf.data.AUTOTUNE


def load_images_from_directory(directory, shuffle=True):
    """Load images and labels from a class-structured directory."""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=RANDOM_SEED,
    )


def resize_and_normalize(images):
    """Normalize pixel values to [0, 1] (images are already loaded at IMAGE_SIZE)."""
    images = tf.cast(images, tf.float32) / 255.0
    return images


def ensure_three_channel_format(images):
    """Ensure image tensors are 3-channel by converting or trimming when needed."""
    channels = images.shape[-1]
    if channels == 1:
        return tf.image.grayscale_to_rgb(images)
    if channels > 3:
        return images[..., :3]
    return images


def get_augmentation_layer():
    """Create training-time data augmentation pipeline."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def preprocess_dataset(dataset, augment=False, augmentation_layer=None):
    """Apply resizing, normalization, channel handling, and optional augmentation."""

    def _preprocess(images, labels):
        images = resize_and_normalize(images)
        images = ensure_three_channel_format(images)
        if augment and augmentation_layer is not None:
            images = augmentation_layer(images, training=True)
        return images, labels

    return dataset.map(_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def create_datasets():
    """Create train, validation, and test TensorFlow datasets."""
    train_raw = load_images_from_directory(DATASET_DIR / "train", shuffle=True)
    val_raw = load_images_from_directory(DATASET_DIR / "val", shuffle=False)
    test_raw = load_images_from_directory(DATASET_DIR / "test", shuffle=False)

    class_names = train_raw.class_names
    augmentation_layer = get_augmentation_layer()

    train_ds = preprocess_dataset(train_raw, augment=True, augmentation_layer=augmentation_layer)
    val_ds = preprocess_dataset(val_raw, augment=False)
    test_ds = preprocess_dataset(test_raw, augment=False)

    return train_ds, val_ds, test_ds, class_names
