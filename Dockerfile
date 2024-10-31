# 使用官方Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt
COPY api/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模型
COPY api/predict_api.py .
COPY onnx_model/model.onnx ./onnx_model/model.onnx

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]