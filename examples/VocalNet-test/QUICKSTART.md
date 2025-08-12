# VocalNet URO-Bench 快速开始指南

本指南将帮助你快速设置并运行VocalNet在URO-Bench上的评估。

## 🚀 快速设置（5分钟）

### 1. 检查环境

确保你有以下模型和数据：

```bash
# 检查模型是否存在
ls /path/to/your/VocalNet-model
ls /path/to/your/CosyVoice2-0.5B-VocalNet

# 检查URO-Bench数据是否存在
ls /path/to/URO-Bench-data
```

### 2. 设置环境变量

```bash
# 设置模型路径
export VOCALNET_MODEL_PATH="/path/to/your/VocalNet-model"
export COSYVOICE_MODEL_PATH="/path/to/your/CosyVoice2-0.5B-VocalNet"

# 设置CUDA设备（可选）
export CUDA_VISIBLE_DEVICES=0
```

### 3. 编辑配置文件

```bash
cd /path/to/URO-Bench
cp scripts/vocalnet-config.sh scripts/my-vocalnet-config.sh
```

编辑 `scripts/my-vocalnet-config.sh`，填入正确的路径：

```bash
# 必须修改的路径
code_dir="/path/to/URO-Bench"
log_dir="/path/to/URO-Bench-log/vocalnet-test"
uro_data_dir="/path/to/URO-Bench-data"
whisper_dir="/path/to/whisper-large-v3"  # 或 "openai/whisper-large-v3"
conda_dir="/path/to/miniconda3/etc/profile.d/conda.sh"

# 环境名称
sdm_env_name="vocalnet"  # 你的VocalNet环境名
uro_env_name="uro"      # URO-Bench环境名

# API密钥（用于评估）
openai_api_key="your-api-key"
gemini_api_key="your-api-key"
```

### 4. 运行测试

```bash
# 激活VocalNet环境
conda activate vocalnet

# 运行快速测试
cd /path/to/URO-Bench
python examples/VocalNet-test/test_vocalnet.py
```

如果测试通过，你会看到：
```
🎉 所有测试通过! VocalNet已成功适配到URO-Bench
```

### 5. 运行评估

```bash
# 运行完整评估
bash scripts/vocalnet-eval.sh scripts/my-vocalnet-config.sh
```

## 🔧 常见问题快速解决

### 问题1: 找不到模型
```
Error: Please set VOCALNET_MODEL_PATH environment variable
```

**解决**:
```bash
export VOCALNET_MODEL_PATH="/absolute/path/to/your/VocalNet-model"
export COSYVOICE_MODEL_PATH="/absolute/path/to/your/CosyVoice2-0.5B-VocalNet"
```

### 问题2: 环境激活失败
```
conda: command not found
```

**解决**: 检查并修正配置文件中的 `conda_dir` 路径：
```bash
# 找到conda路径
which conda
# 应该类似: /home/user/miniconda3/bin/conda
# 那么conda_dir应该设置为: /home/user/miniconda3/etc/profile.d/conda.sh
```

### 问题3: CUDA内存不足
```
RuntimeError: CUDA out of memory
```

**解决**: 修改推理代码中的batch size或使用CPU：
```bash
export CUDA_VISIBLE_DEVICES=""  # 使用CPU
```

### 问题4: 依赖缺失
```
ModuleNotFoundError: No module named 'xxx'
```

**解决**: 安装缺失的依赖：
```bash
conda activate vocalnet
pip install torchaudio librosa soundfile whisper
pip install hyperpyyaml onnxruntime jsonlines
```

## 📊 快速验证单个数据集

如果想快速测试单个数据集：

```bash
# 测试单轮对话
conda activate vocalnet
python examples/VocalNet-test/inference_for_eval.py \
    --dataset /path/to/URO-Bench-data/basic/AlpacaEval/test.jsonl \
    --output_dir ./quick_test

# 测试多轮对话
python examples/VocalNet-test/inference_multi.py \
    --dataset /path/to/URO-Bench-data/pro/MtBenchEval-en/test.jsonl \
    --output_dir ./quick_test_multi
```

## 🎯 选择性评估

如果只想评估部分数据集，编辑配置文件中的 `datasets` 数组：

```bash
# 只评估几个基础数据集
datasets=(
    "AlpacaEval 199 open basic en"
    "Repeat 252 wer basic en"
)

# 注释掉不需要的数据集
# "CommonEval 200 open basic en"
# "WildchatEval 349 open basic en"
```

## 📈 查看结果

评估完成后，结果保存在 `log_dir` 中：

```bash
# 查看总体结果
cat /path/to/log_dir/eval/summary.json

# 查看具体数据集结果
ls /path/to/log_dir/eval/basic/
ls /path/to/log_dir/eval/pro/
```

## 💡 性能优化建议

1. **使用本地Whisper模型**：避免网络下载延迟
2. **设置合适的GPU**：使用 `CUDA_VISIBLE_DEVICES` 选择GPU
3. **调整模型参数**：在推理代码中修改 `temperature`, `top_p` 等参数
4. **选择性评估**：先测试小数据集，确认无误后再运行全部

## 📞 获取帮助

如果遇到问题：

1. 运行测试脚本获取详细错误信息
2. 检查日志文件
3. 确认所有路径和环境变量设置正确
4. 查看完整的 README.md 文档
