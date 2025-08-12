# VocalNet URO-Bench 适配说明

本目录包含了将VocalNet模型适配到URO-Bench评估框架的完整代码。

## 文件结构

```
VocalNet-test/
├── inference_for_eval.py    # 单轮对话推理脚本
├── inference_multi.py       # 多轮对话推理脚本
├── README.md               # 本说明文档
└── omni_speech/           # VocalNet模型代码
    └── infer/
        └── vocalnet.py    # VocalNet推理类
```

## 环境准备

### 1. 安装依赖

首先确保你有两个conda环境：

```bash
# URO-Bench环境（用于评估）
conda create -n uro python=3.11
conda activate uro
pip install -r /path/to/URO-Bench/requirements.txt

# VocalNet环境（用于模型推理）
conda create -n vocalnet python=3.11
conda activate vocalnet
# 安装VocalNet相关依赖
pip install torch torchaudio transformers
pip install whisper librosa soundfile
pip install hyperpyyaml onnxruntime
# 其他VocalNet需要的依赖...
```

### 2. 模型准备

下载并准备以下模型：

#### VocalNet模型
```bash
# 下载VocalNet模型权重
# 设置模型路径到环境变量
export VOCALNET_MODEL_PATH="/path/to/your/VocalNet-model"
```

#### CosyVoice模型
```bash
# 下载CosyVoice2-0.5B-VocalNet模型
# 设置模型路径到环境变量
export COSYVOICE_MODEL_PATH="/path/to/your/CosyVoice2-0.5B-VocalNet"
```

#### Whisper模型（用于ASR评估）
```bash
# 方法1：使用在线模型（需要稳定网络）
whisper_dir="openai/whisper-large-v3"

# 方法2：下载本地模型（推荐）
huggingface-cli download --resume-download openai/whisper-large-v3 --local-dir ./whisper-large-v3
whisper_dir="/path/to/whisper-large-v3"
```

### 3. 数据准备

```bash
# 下载URO-Bench数据
cd /path/to/URO-Bench/..
export HF_ENDPOINT=https://hf-mirror.com  # 如果网络有问题
huggingface-cli download --repo-type dataset --resume-download Honggao/URO-Bench URO-Bench-data.zip --local-dir ./ --local-dir-use-symlinks False
unzip URO-Bench-data.zip
```

## 配置设置

### 1. 编辑配置文件

编辑 `scripts/vocalnet-config.sh`，设置正确的路径：

```bash
# 必须设置的路径
export VOCALNET_MODEL_PATH="/path/to/your/VocalNet-model"
export COSYVOICE_MODEL_PATH="/path/to/your/CosyVoice2-0.5B-VocalNet"
code_dir="/path/to/URO-Bench"
log_dir="/path/to/URO-Bench-log/vocalnet-test"
uro_data_dir="/path/to/URO-Bench-data"
whisper_dir="/path/to/whisper-large-v3"
conda_dir="/path/to/miniconda3/etc/profile.d/conda.sh"
sdm_env_name="vocalnet"
uro_env_name="uro"

# API密钥（用于某些评估）
openai_api_key="your-openai-api-key"
gemini_api_key="your-gemini-api-key"
```

### 2. 验证配置

确保以下路径存在且可访问：
- VocalNet模型文件
- CosyVoice模型文件
- URO-Bench数据目录
- Whisper模型文件
- Conda环境

## 运行评估

### 完整评估

运行完整的URO-Bench评估：

```bash
cd /path/to/URO-Bench
bash scripts/vocalnet-eval.sh scripts/vocalnet-config.sh
```

### 单独测试

#### 测试单轮对话推理

```bash
cd /path/to/URO-Bench
conda activate vocalnet

python examples/VocalNet-test/inference_for_eval.py \
    --dataset /path/to/URO-Bench-data/basic/AlpacaEval/test.jsonl \
    --output_dir ./test_output
```

#### 测试多轮对话推理

```bash
cd /path/to/URO-Bench
conda activate vocalnet

python examples/VocalNet-test/inference_multi.py \
    --dataset /path/to/URO-Bench-data/pro/MtBenchEval-en/test.jsonl \
    --output_dir ./test_multi_output
```

## 结果查看

评估完成后，结果将保存在配置的`log_dir`中：

```
log_dir/
├── eval/
│   ├── basic/          # 基础轨道结果
│   │   ├── AlpacaEval/
│   │   ├── CommonEval/
│   │   └── ...
│   ├── pro/            # 专业轨道结果
│   │   ├── MtBenchEval-en/
│   │   ├── GenEmotion-en/
│   │   └── ...
│   └── summary.json    # 总体评估结果
```

每个数据集的结果包含：
- `pred_text.jsonl`: 模型预测的文本
- `question_text.jsonl`: 输入问题文本
- `gt_text.jsonl`: 参考答案文本
- `audio/`: 生成的音频文件
- `eval_with_asr/`: ASR和评估结果

## 故障排除

### 常见问题

1. **模型路径错误**
   ```
   Error: Please set VOCALNET_MODEL_PATH environment variable
   ```
   解决：确保在配置文件中正确设置了模型路径

2. **环境激活失败**
   ```
   conda: command not found
   ```
   解决：检查`conda_dir`路径是否正确

3. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：减少batch size或使用更小的模型

4. **音频文件生成失败**
   ```
   Warning: No audio generated
   ```
   解决：检查CosyVoice模型路径和权重文件

### 调试模式

添加详细日志输出：

```bash
export PYTHONPATH="/path/to/URO-Bench/examples/VocalNet-test:$PYTHONPATH"
python -u examples/VocalNet-test/inference_for_eval.py --dataset ... --output_dir ...
```

## 自定义配置

### 修改模型参数

在`inference_for_eval.py`和`inference_multi.py`中的`load_sdm()`函数中修改：

```python
vocalnet_model = VocalNetModel(
    model_name_or_path=VOCALNET_MODEL_PATH,
    vocoder_path=COSYVOICE_MODEL_PATH,
    s2s=True,
    temperature=0.0,        # 调整温度参数
    num_beams=1,           # 调整beam search
    max_new_tokens=512,    # 调整最大生成长度
    top_p=0.1,             # 调整top-p采样
    streaming=False
)
```

### 选择评估数据集

在配置文件中注释/取消注释要评估的数据集：

```bash
datasets=(
    "AlpacaEval 199 open basic en"      # 取消注释要测试的数据集
    # "CommonEval 200 open basic en"   # 注释掉不需要的数据集
    # ...
)
```

## 性能参考

VocalNet在URO-Bench上的预期性能（待测试）：

| Track | Language | Overall | UTMOS | ASR-WER/CER |
|-------|----------|---------|-------|-------------|
| Basic | EN       | TBD     | TBD   | TBD         |
| Basic | ZH       | TBD     | TBD   | TBD         |
| Pro   | EN       | TBD     | TBD   | TBD         |
| Pro   | ZH       | TBD     | TBD   | TBD         |

## 贡献

如果你在使用过程中发现问题或有改进建议，欢迎提交Issue或Pull Request。

## 许可

本适配代码遵循MIT许可证。请确保遵守VocalNet和CosyVoice的相关许可证条款。