from tf_keras import layers, models
import numpy as np
import cv2
from tf_keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow as tf
import os
import multiprocessing
from tf_keras.applications.resnet_v2 import ResNet50V2
from tf_keras.applications.resnet_v2 import preprocess_input
import datetime
from PIL import Image
from numpy import expand_dims

positive_dir = "positive"
negative_dir = "negative"

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

# 创建模型
def create_model(input_shape=(224, 224, 3)):
    # 加载预训练的ResNet50V2模型，不包括顶层
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 冻结基础模型的权重
    base_model.trainable = False

    # 建新的顶层
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# 创建数据生成器
def create_data_generator(directory, target_size=(224, 224), batch_size=32):
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
    ).flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # 添加单个图片的预处理函数
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def verify_image_sizes(directory, target_size=(224, 224)):
    """验证目录中的图片尺寸"""
    irregular_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as img:
                        if img.size != target_size:
                            irregular_images.append((image_path, img.size))
                except Exception as e:
                    print(f"无法打开图片 {image_path}: {str(e)}")

    return irregular_images

def train_model(train_dir, batch_size=32, epochs=50, target_size=(224, 224),
                checkpoint_dir='checkpoints', max_checkpoints=5,
                resume_from=None, initial_epoch=0):
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    print("正在验证训练数据集中的图片尺寸...")
    irregular_images = verify_image_sizes(train_dir, target_size)

    if irregular_images:
        print("\n发现以下图片尺寸不规范：")
        for path, size in irregular_images:
            print(f"图片: {path}, 尺寸: {size}")
        print(f"\n所有图片将被自动调整为 {target_size}")

    # 创建数据生成器
    train_generator = create_data_generator(
        train_dir,
        target_size=target_size,
        batch_size=batch_size
    )

    # 验证集生成器
    validation_generator = create_data_generator(
        train_dir,
        target_size=target_size,
        batch_size=batch_size
    )

    # 修改数据加载部分
    def load_data_safely(generator):
        images_list = []
        labels_list = []

        n_samples = generator.samples
        steps = n_samples // generator.batch_size + (1 if n_samples % generator.batch_size else 0)

        for _ in range(steps):
            try:
                images, labels = next(generator)
                images_list.append(images)
                labels_list.append(labels)
            except StopIteration:
                break

        if images_list and labels_list:
            return np.concatenate(images_list), np.concatenate(labels_list)
        return np.array([]), np.array([])

    print("正在将数据加载到内存中...")

    # 加载训练数据
    train_images, train_labels = load_data_safely(train_generator)
    val_images, val_labels = load_data_safely(validation_generator)

    if len(train_images) == 0 or len(val_images) == 0:
        raise ValueError("无法加载数据，请检查数据集")

    # 确保批次大小不大于数据集大小
    batch_size = min(batch_size, len(train_images))

    print(f"数据加载完成。训练集大小: {len(train_images)} 验证集大小: {len(val_images)}")

    # 在数据加载后添加验证步骤
    print("\n数据集信息验证:")
    print(f"训练集类别分布:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"类别 {label}: {count} 样本")

    print(f"\n验证集类别分布:")
    unique, counts = np.unique(val_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"类别 {label}: {count} 样本")

    print(f"\n数据范围检查:")
    print(f"训练集数据范围: [{np.min(train_images):.2f}, {np.max(train_images):.2f}]")
    print(f"验证集数据范围: [{np.min(val_images):.2f}, {np.max(val_images):.2f}]")

    # 创建或加载模型
    if resume_from and os.path.exists(resume_from):
        print(f"从checkpoint {resume_from} 恢复训练...")
        model = models.load_model(resume_from)
    else:
        print("创建新模型...")
        model = create_model()

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        'model_epoch_{epoch:03d}_val_acc_{val_accuracy:.4f}.keras'
    )

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=False,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # 每个epoch记录直方图
            write_graph=True,  # 记录模型图
            write_images=True,  # 记录模型权重
            update_freq='epoch',  # 每个epoch更新
            profile_batch=2,  # 记录性能分析
        )
    ]

    history = model.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        shuffle=True  # 添加随机打乱
    )

    return model, history

# 预测函数
def predict_image(model, image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return bool(prediction > 0.5)

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的checkpoint文件"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.keras'):
            filepath = os.path.join(checkpoint_dir, file)
            created_time = os.path.getctime(filepath)
            checkpoints.append((filepath, created_time))

    if not checkpoints:
        return None

    # 按创建时间排序并返回最新的
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints[0][0]

if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(multiprocessing.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(multiprocessing.cpu_count())

    # 设置训练参数
    BATCH_SIZE = 12
    EPOCHS = 50
    TRAIN_DIR = "./dataset"  # 修改为实际的数据集根目录
    CHECKPOINT_DIR = "checkpoints"
    MAX_CHECKPOINTS = 5
    RESUME_FROM = find_latest_checkpoint(CHECKPOINT_DIR)
    if RESUME_FROM:
        print(f"找到最新的模型文: {RESUME_FROM}")
        # 获取最后训练的epoch数
        epoch_num = int(RESUME_FROM.split('epoch_')[1].split('_')[0])
        initial_epoch = epoch_num
        print(f"将从epoch {initial_epoch} 继续训练")
    else:
        print("未找到现有模型，将开始新的训练")
        initial_epoch = 0

    # 验证数据集目录结构
    required_dirs = ['positive', 'negative']
    total_images = 0
    for dir_name in required_dirs:
        dir_path = os.path.join(TRAIN_DIR, dir_name)
        if not os.path.exists(dir_path):
            raise ValueError(f"找不到必需的目录: {dir_path}")

        # 检查目录中是否有图片文件
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            raise ValueError(f"目录 {dir_path} 中没有找到图片文件")
        total_images += len(image_files)
        print(f"在 {dir_name} 目录中找到 {len(image_files)} 个图片文件")

    if total_images < 100:  # 设置最小数据集大小
        print("警告：数据集样本数量过少")

    # 开始训练
    print("开始训练模型...")
    model, history = train_model(
        TRAIN_DIR,
        BATCH_SIZE,
        EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        max_checkpoints=MAX_CHECKPOINTS,
        resume_from=RESUME_FROM,
        initial_epoch=initial_epoch
    )

    # 保存最终模型
    final_model_path = os.path.join(CHECKPOINT_DIR, 'final_model.keras')
    model.save(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")

    # 打印训练结果
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"最终训练准确率: {final_accuracy:.4f}")
    print(f"最终验证准确率: {final_val_accuracy:.4f}")
    print(f"可以使用以下命令启动TensorBoard查看训练过程：")
    print(f"tensorboard --logdir=logs")