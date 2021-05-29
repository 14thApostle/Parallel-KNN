import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import shutil
import argparse

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-dir", type=str, required=True, help="Root path to the dataset"
)
parser.add_argument(
    "--batch-size", type=int, default=16, help="Batch size for all GPUs"
)
parser.add_argument(
    "--plt-dataset",
    action="store_true",
    help="Pass arg to plot dataset examples",
)
parser.add_argument(
    "--img-size",
    nargs="+",
    type=int,
    default=[180, 180],
    help="[train, test] image sizes in format [IMG_W, IMG_H]",
)
parser.add_argument(
    "--no-plts",
    action="store_true",
    help="Pass arg to not plot any figures",
)

opt = parser.parse_args()

root_dataset_dir = os.path.abspath(opt.dataset_dir)
train_dir = join(root_dataset_dir, "train")
test_dir = join(root_dataset_dir, "test")

batch_size = opt.batch_size
if len(opt.img_size) > 2:
    raise Exception("Invalid Image size")

img_height = opt.img_size[0]
img_width = opt.img_size[-1]

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names
print(class_names)

if opt.plt_dataset and not opt.no_plts:
    utils.show_dataset(train_ds, class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.experimental.preprocessing.Rescaling(1.0 / 255)

num_classes = 2

model = Sequential(
    [
        layers.experimental.preprocessing.Rescaling(
            1.0 / 255, input_shape=(img_height, img_width, 3)
        ),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu", name="ftr_layer"),
        layers.Dense(num_classes),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()

epochs = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

if not opt.no_plts:
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, seed=123, image_size=(img_height, img_width), batch_size=batch_size
)

print("Results on test dataset")
model.evaluate(test_ds)

feature_xtr = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="ftr_layer").output,
)

root_path = root_dataset_dir + "_ftrs"
if os.path.exists(root_path):
    shutil.rmtree(root_path)
os.makedirs(root_path)
os.makedirs(join(root_path, "train"))
os.makedirs(join(root_path, "test"))

## Train
correct = 0
total = 0

f1 = open(join(root_path, "train" + ".txt"), "w")
for class_name in os.listdir(train_dir):
    print("Converting train", class_name, "images")
    label = class_names.index(class_name)
    for img_path in os.listdir(join(train_dir, class_name)):
        img_path = join(train_dir, class_name, img_path)
        img = np.array(Image.open(img_path).resize((img_height, img_width)))
        predictions = model.predict(tf.expand_dims(img, axis=0))
        scores = tf.nn.softmax(predictions, axis=1)
        output = tf.math.argmax(scores, axis=1)[0]

        ftr_vector = feature_xtr.predict(tf.expand_dims(img, axis=0))
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        path = join(root_path, "train", img_id + ".txt")
        f = open(path, "w")
        f.write(
            " ".join(
                [str(i) for i in [img_id, str(class_names.index(class_name)), len(ftr_vector.flatten()), "\n"]]
            )
        )
        for val in ftr_vector.flatten():
            f.write(str(val) + " ")
        f.close()

        for val in ftr_vector.flatten():
            f1.write(str(val) + " ")
        f1.write(str(class_names.index(class_name)) + "\n")

        if output == label:
            correct += 1
        total += 1
f1.close()
print("Train accuracy", correct / total)

## Test
correct = 0
total = 0
f1 = open(join(root_path, "test" + ".txt"), "w")
for class_name in os.listdir(test_dir):
    print("Converting test", class_name, "images")
    label = class_names.index(class_name)
    for img_path in os.listdir(join(test_dir, class_name)):
        img_path = join(test_dir, class_name, img_path)
        img = np.array(Image.open(img_path).resize((img_height, img_width)))
        predictions = model.predict(tf.expand_dims(img, axis=0))
        scores = tf.nn.softmax(predictions, axis=1)
        output = tf.math.argmax(scores, axis=1)[0]

        ftr_vector = feature_xtr.predict(tf.expand_dims(img, axis=0))
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        path = join(root_path, "test", img_id + ".txt")
        f = open(path, "w")
        f.write(
            " ".join(
                [str(i) for i in [img_id, class_name, len(ftr_vector.flatten()), "\n"]]
            )
        )
        for val in ftr_vector.flatten():
            f.write(str(val) + " ")
        f.close()

        for val in ftr_vector.flatten():
            f1.write(str(val) + " ")
        f1.write(str(class_names.index(class_name)) + "\n")

        if output == label:
            correct += 1
        total += 1
f1.close()
print("Test accuracy", correct / total)
