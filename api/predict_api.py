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

        # 尝试直接解码图像
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            # 如果是静态图片（PNG, JPEG, BMP等）
            return {"type": "static", "data": np.expand_dims(_process_single_frame(img, target_size=target_size), axis=0)}

        # 如果直接解码失败，尝试作为GIF处理
        frames = []
        try:
            temp_path = bytes_to_video_path(image_data)
            cap = cv2.VideoCapture(temp_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(_process_single_frame(frame, target_size=target_size))
            cap.release()

            if frames:
                return {"type": "gif", "data": np.array(frames)}
        except Exception as gif_error:
            logging.warning(f"GIF处理失败: {str(gif_error)}")

        raise ValueError("不支持的图像格式或图像损坏")

    except Exception as e:
        logging.error(f"图像预处理失败: {str(e)}")
        raise

def _process_single_frame(img, target_size=(224, 224)):
    """处理单帧图像"""
    try:
        # 确保图像是3通道RGB
        if len(img.shape) == 2:  # 灰度图
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA图
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整图像大小
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # 标准化
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (img - mean) / std
    except Exception as e:
        logging.error(f"帧处理失败: {str(e)}")
        raise

def bytes_to_video_path(image_data):
    """将图像字节数据保存为临时文件"""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    temp.write(image_data)
    temp.close()
    return temp.name

@app.post("/predict")
async def predict_image(file: UploadFile):
    """接收图片并返回预测结果"""
    # 记录请求信息
    logging.info(f"收到预测请求 - 文件名: {file.filename}, 文件类型: {file.content_type}, 文件大小: {file.size} bytes")

    allowed_types = {'image/jpeg', 'image/png', 'image/gif',
                    'image/bmp', 'image/webp', 'image/tiff'}

    if not (file.content_type in allowed_types):
        error_msg = f"不支持的文件类型。支持的类型: {', '.join(allowed_types)}"
        logging.warning(f"请求被拒绝 - {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )

    try:
        image_data = await file.read()
        processed_result = preprocess_image(image_data)

        # 记录处理类型
        logging.info(f"图像处理类型: {processed_result['type']}")

        if processed_result["type"] == "static":
            # 处理静态图片
            predictions = global_session.run(None, {global_input_name: processed_result["data"]})[0]
            probability = float(predictions[0][0])
            result = {
                "type": "static",
                "result": probability > 0.5,
                "probability": probability
            }
            logging.info(f"预测完成 - 结果: {result}")
            return result
        else:
            # 处理GIF
            max_probability = 0.0
            frame_count = processed_result["data"].shape[0]

            for frame in processed_result["data"]:
                frame_expanded = np.expand_dims(frame, axis=0)
                predictions = global_session.run(None, {global_input_name: frame_expanded})[0]
                probability = float(predictions[0][0])
                max_probability = max(max_probability, probability)

                # 如果任何一帧检测到目标，立即返回
                if probability > 0.5:
                    return {
                        "type": "gif",
                        "result": True,
                        "probability": probability,
                        "frame_count": frame_count
                    }

            return {
                "type": "gif",
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