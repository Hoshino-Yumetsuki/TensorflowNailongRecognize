import tensorflow as tf
from tf_keras import models
import os
import tf2onnx

def convert_to_pb(model_path, output_dir='saved_model'):
    """
    将Keras模型转换为SavedModel(pb)格式

    参数:
        model_path: str, .keras模型文件的路径
        output_dir: str, 输出目录路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 加载模型
        print(f"正在加载模型: {model_path}")
        model = models.load_model(model_path)

        # 打印模型信息
        print("\n模型信息:")
        print(f"输入形状: {model.inputs[0].shape}")
        print(f"输出形状: {model.outputs[0].shape}")

        # 转换并保存模型
        print(f"\n正在将模型转换为SavedModel格式...")
        tf.saved_model.save(model, output_dir)
        print(f"模型已保存到: {output_dir}")

        # 验证保存的模型
        print("\n正在验证保存的模型...")
        loaded = tf.saved_model.load(output_dir)
        print("模型验证成功!")

        # 打印签名信息
        print("\n模型签名信息:")
        print(loaded.signatures.keys())

        return True

    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        return False

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

def convert_to_frozen_graph(saved_model_dir, output_graph_path):
    """
    将SavedModel转换为冻结图（Frozen Graph）

    参数:
        saved_model_dir: str, SavedModel目录路径
        output_graph_path: str, 输出的冻结图路径
    """
    try:
        print(f"正在将SavedModel转换为冻结图...")

        # 加载SavedModel
        model = tf.saved_model.load(saved_model_dir)
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        # 获取冻结图
        frozen_func = tf.function(lambda x: concrete_func(x))
        frozen_func = frozen_func.get_concrete_function(
            tf.TensorSpec(concrete_func.inputs[0].shape, concrete_func.inputs[0].dtype))

        # 将函数转换为图
        graph_def = frozen_func.graph.as_graph_def()

        # 移除训练相关的节点（使用新的方法）
        output_node_names = [node.name for node in graph_def.node if node.op == 'Identity']
        converted_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=None,
            input_graph_def=graph_def,
            output_node_names=output_node_names
        )

        # 将图写入文件
        tf.io.write_graph(converted_graph_def,
                         os.path.dirname(output_graph_path),
                         os.path.basename(output_graph_path),
                         as_text=False)

        print(f"冻结图已保存到: {output_graph_path}")
        return True

    except Exception as e:
        print(f"转换冻结图时发生错误: {str(e)}")
        return False

def convert_to_onnx(model_path, output_path='model.onnx'):
    """
    将Keras模型转换为ONNX格式

    参数:
        model_path: str, .keras模型文件的路径
        output_path: str, 输出的ONNX文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 加载模型
        print(f"正在加载模型: {model_path}")
        model = models.load_model(model_path)

        # 转换为ONNX
        print(f"\n正在将模型转换为ONNX格式...")
        spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
        output_path = output_path.replace('\\', '/')  # 确保路径格式正确

        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
        print(f"模型已保存到: {output_path}")

        return True

    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 设置路径
    CHECKPOINT_DIR = "checkpoints"
    ONNX_PATH = "onnx_model/model.onnx"

    # 创建输出目录
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

    # 查找最新的模型文件
    model_path = find_latest_checkpoint(CHECKPOINT_DIR)

    if model_path is None:
        print("错误: 未找到模型文件!")
        exit(1)

    print(f"找到最新的模型文件: {model_path}")

    # 转换为ONNX格式
    if convert_to_onnx(model_path, ONNX_PATH):
        print("\n模型转换完成!")
        print(f"ONNX模型保存在: {ONNX_PATH}")
    else:
        print("\n模型转换失败!")