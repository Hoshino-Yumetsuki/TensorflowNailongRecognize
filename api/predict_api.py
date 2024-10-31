import logging
from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import onnxruntime as ort
import cv2
import tempfile
from contextlib import asynccontextmanager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('api_prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理"""
    MODEL_PATH = "onnx_model/model.onnx"
    load_model(MODEL_PATH)
    yield

app = FastAPI(lifespan=lifespan)

# 全局变量存储模型会话
global_session = None
global_input_name = None

def load_model(model_path: str):
    """加载ONNX模型"""
    global global_session, global_input_name
    try:
        global_session = ort.InferenceSession(model_path)
        global_input_name = global_session.get_inputs()[0].name
        logging.info("ONNX模型加载成功")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        raise

def preprocess_image(image_data, target_size=(224, 224)):
    """预处理图像数据"""
    try:
        # 将图像数据转换为numpy数组
        nparr = np.frombuffer(image_data, np.uint8)

        # 创建内存缓冲区用于读取GIF
        buf = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if buf is None:
            raise ValueError("无法解码图像")

        # 如果是静态图片，直接处理
        if len(buf.shape) == 3:
            return np.expand_dims(_process_single_frame(buf), axis=0)

        # 如果是GIF，处理所有帧
        frames = []
        cap = cv2.VideoCapture(bytes_to_video_path(image_data))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(_process_single_frame(frame))
        cap.release()

        if not frames:
            raise ValueError("无法提取图像帧")

        return np.array(frames)

    except Exception as e:
        logging.error(f"图像预处理失败: {str(e)}")
        raise

def _process_single_frame(img, target_size=(224, 224)):
    """处理单帧图像"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img - mean) / std

def bytes_to_video_path(image_data):
    """将图像字节数据保存为临时文件"""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    temp.write(image_data)
    temp.close()
    return temp.name

@app.post("/predict")
async def predict_image(file: UploadFile):
    """
    接收图片并返回预测结果
    返回格式: {"result": bool, "probability": float, "frame_count": int}
    """
    if not (file.content_type.startswith('image/')):
        raise HTTPException(status_code=400, detail="只接受图片文件")

    try:
        image_data = await file.read()
        processed_images = preprocess_image(image_data)

        # 对每一帧进行预测
        max_probability = 0.0
        frame_count = processed_images.shape[0]

        for frame in processed_images:
            frame_expanded = np.expand_dims(frame, axis=0)
            predictions = global_session.run(None, {global_input_name: frame_expanded})[0]
            probability = float(predictions[0][0])
            max_probability = max(max_probability, probability)

            # 如果任何一帧检测到目标，立即返回
            if probability > 0.5:
                return {
                    "result": True,
                    "probability": probability,
                    "frame_count": frame_count
                }

        return {
            "result": False,
            "probability": max_probability,
            "frame_count": frame_count
        }

    except Exception as e:
        logging.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)