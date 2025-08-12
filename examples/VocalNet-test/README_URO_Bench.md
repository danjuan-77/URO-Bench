# VocalNet URO-Bench 适配指南

这个目录包含了将VocalNet模型适配到URO-Bench评估框架的完整实现。

## 文件说明

### 核心文件
- `inference_for_eval.py` - 单轮对话推理脚本
- `inference_multi.py` - 多轮对话推理脚本
- `omni_speech/infer/vocalnet.py` - VocalNet模型推理代码（已修改支持环境变量）

### 配置和脚本
- `../../scripts/vocalnet-config.sh` - 配置文件模板
- `../../scripts/vocalnet-eval.sh` - 主要评估脚本
- `../../scripts/vocalnet-asr-eval.sh` - 完整评估脚本（独立版本）

## 安装和配置

### 1. 环境准备

```bash
# 创建VocalNet环境
conda create -n vocalnet python=3.11
conda activate vocalnet

# 安装VocalNet依赖
cd examples/VocalNet-test
pip install -r requirements.txt

# 确保也有URO-Bench环境
conda create -n uro python=3.11
conda activate uro
cd ../../
pip install -r requirements.txt
```

### 2. 模型准备

你需要准备以下模型：

1. **VocalNet模型** - VocalNet的主要模型权重
2. **CosyVoice2-0.5B模型** - 用于语音合成的vocoder模型
3. **Whisper-large-v3** - 用于ASR评估（可选，如果网络正常可以使用`openai/whisper-large-v3`）

### 3. 配置文件设置

复制并修改配置文件：

```bash
cp scripts/vocalnet-config.sh scripts/my-vocalnet-config.sh
```

编辑 `scripts/my-vocalnet-config.sh`，填入正确的路径：

```bash
# 模型路径配置（必需）
export VOCALNET_MODEL="/path/to/your/vocalnet/model"           # VocalNet模型路径
export COSYVOICE_MODEL="/path/to/your/cosyvoice2-0.5b/model"  # CosyVoice2-0.5B模型路径

# 目录配置（必需）
code_dir="/path/to/URO-Bench"                                 # URO-Bench代码目录
log_dir="/path/to/URO-Bench-log/VocalNet-test"               # 评估结果保存目录
whisper_dir="/path/to/whisper-large-v3"                      # whisper模型路径
uro_data_dir="/path/to/URO-Bench-data"                       # URO-Bench数据目录

# conda环境配置（必需）
conda_dir="/path/to/miniconda3/etc/profile.d/conda.sh"       # conda路径
sdm_env_name="vocalnet"                                       # VocalNet环境名称
uro_env_name="uro"                                            # URO-Bench环境名称

# API密钥配置（评估阶段需要）
openai_api_key="your-openai-api-key"                         # OpenAI API密钥
gemini_api_key="your-gemini-api-key"                         # Gemini API密钥
```

### 4. 数据准备

按照URO-Bench主README的说明下载数据：

```bash
cd ..  # 到URO-Bench根目录
export HF_ENDPOINT=https://hf-mirror.com  # 如果网络有问题
huggingface-cli download --repo-type dataset --resume-download Honggao/URO-Bench URO-Bench-data.zip --local-dir ./ --local-dir-use-symlinks False
unzip URO-Bench-data.zip
```

## 使用方法

### 快速开始

1. **配置完成后，运行完整评估：**

```bash
bash scripts/vocalnet-eval.sh scripts/my-vocalnet-config.sh
```

### 单独测试

1. **测试单个数据集：**

```bash
# 激活VocalNet环境
conda activate vocalnet

# 设置环境变量
export VOCALNET_MODEL="/path/to/your/vocalnet/model"
export COSYVOICE_MODEL="/path/to/your/cosyvoice2-0.5b/model"

# 运行推理
cd examples/VocalNet-test
python inference_for_eval.py \
    --dataset /path/to/URO-Bench-data/basic/AlpacaEval/test.jsonl \
    --output_dir /path/to/output
```

2. **测试多轮对话：**

```bash
python inference_multi.py \
    --dataset /path/to/URO-Bench-data/pro/MtBenchEval-en/test.jsonl \
    --output_dir /path/to/output
```

## 评估流程

评估流程包括以下步骤：

1. **模型推理** - 使用VocalNet生成文本和音频响应
2. **ASR转录** - 使用Whisper对生成的音频进行转录
3. **自动评分** - 根据不同的评估模式进行评分：
   - `open` - 开放式问题，使用OpenAI API评估
   - `semi-open` - 半开放式问题，使用参考答案和API评估
   - `qa` - 问答题，使用精确匹配或相似度评估
   - `wer` - 重复任务，使用词错率评估
   - 其他专门任务（如情感生成、说话人识别等）

## 输出结果

评估完成后，结果将保存在 `${log_dir}/eval` 目录下：

- 每个数据集的详细结果在对应的子目录中
- `evaluate.py` 会生成总体评估报告
- 音频输出保存在各数据集的 `audio` 子目录中

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查 `VOCALNET_MODEL` 和 `COSYVOICE_MODEL` 路径是否正确
   - 确保模型文件完整且可访问

2. **环境依赖问题**
   - 确保所有依赖包都已正确安装
   - 检查CUDA版本是否兼容

3. **音频生成失败**
   - 检查CosyVoice模型是否正确加载
   - 确保有足够的GPU内存

4. **API调用失败**
   - 检查OpenAI和Gemini API密钥是否有效
   - 确保网络连接正常

### 调试模式

如果遇到问题，可以：

1. 在推理脚本中添加更多日志输出
2. 使用较小的数据集进行测试
3. 检查中间结果文件的内容

## 定制化

### 修改评估参数

可以在配置文件中修改要评估的数据集列表：

```bash
# 只评估英文基础数据集
datasets=(
    "AlpacaEval 199 open basic en"
    "CommonEval 200 open basic en"
    "Repeat 252 wer basic en"
)
```

### 修改模型参数

可以在 `load_sdm()` 函数中修改VocalNet的推理参数：

```python
vocalnet = VocalNetModel(
    model_name_or_path=vocalnet_model_path,
    vocoder_path=cosyvoice_model_path,
    s2s=True,
    temperature=0.1,      # 调整采样温度
    num_beams=1,          # 调整beam search
    max_new_tokens=1024,  # 调整最大生成长度
    top_p=0.1,           # 调整nucleus sampling
    streaming=False
)
```

## 贡献

如果你发现问题或有改进建议，请：

1. 检查问题是否已知
2. 提供详细的错误信息和环境配置
3. 如果可能，提供修复建议

## 许可证

此适配代码遵循URO-Bench的MIT许可证。
