import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --------------------------
# 1. 加载与归一化
# --------------------------
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# --------------------------
# 2. 数据增强
# --------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# --------------------------
# 3. 改进版 CNN
# --------------------------
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    data_augmentation,

    # Conv Block 1
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),

    # Conv Block 2
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),

    # Conv Block 3
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),

    # 全连接层
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# --------------------------
# 4. 编译
# --------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------
# 5. 训练
# --------------------------
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=20,
    batch_size=64
)

# --------------------------
# 6. 可视化结果
# --------------------------
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)