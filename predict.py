import os
import logging
import cv2
import numpy as np
import onnxruntime as ort

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',  # 简化日志格式
    handlers=[
        logging.FileHandler('prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图像")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)

        # 明确指定 float32 类型
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        return img.astype(np.float32)  # 确保返回float32类型
    except Exception as e:
        logging.error(f"图像预处理失败: {str(e)}")
        raise

def predict_folder(model_path, image_folder):
    """
    对指定文件夹中的所有图片进行预测
    """
    # 加载 ONNX 模型
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        logging.info("ONNX模型加载成功")

        # 获取输入形状
        input_shape = session.get_inputs()[0].shape
        logging.info(f"模型期望的输入形状: {input_shape}")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        return

    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_folder)
                  if f.lower().endswith(valid_extensions)]

    if not image_files:
        logging.warning(f"在 {image_folder} 中没有找到图片")
        return

    # 预测每张图片
    for image_file in image_files:
        # 使用 os.path.abspath 获取绝对路径，并确保路径编码正确
        image_path = os.path.abspath(os.path.join(image_folder, image_file))
        try:
            img = preprocess_image(image_path)
            # 确保输入数据类型为float32
            img = np.expand_dims(img, axis=0).astype(np.float32)

            # 添加类型检查日志
            logging.info(f"输入数据类型: {img.dtype}")

            # ONNX推理
            predictions = session.run(None, {input_name: img})[0]
            prediction = float(predictions[0][0])
            result = prediction > 0.5

            logging.info(f"{image_file}: {result} (概率: {prediction:.2%})")

        except Exception as e:
            logging.error(f"处理 {image_file} 失败: {str(e)}")

if __name__ == "__main__":
    MODEL_PATH = "onnx_model/model.onnx"    # 修改为ONNX模型路径
    IMAGE_FOLDER = "./test_images"

    if not os.path.exists(IMAGE_FOLDER):
        logging.error(f"图片文件夹不存在: {IMAGE_FOLDER}")
    else:
        predict_folder(MODEL_PATH, IMAGE_FOLDER)
