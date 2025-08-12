#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# VocalNet模型配置 - 请根据实际情况修改这些路径
model_name=VocalNet

# 模型路径配置（必需）
export VOCALNET_MODEL="/home/tuwenming/Models/VocalNet/VocalNet-1B/"                                # VocalNet模型路径，例如：/path/to/VocalNet-model
export COSYVOICE_MODEL="/home/tuwenming/Models/FunAudioLLM/CosyVoice2-0.5B"                               # CosyVoice2-0.5B模型路径，例如：/path/to/CosyVoice2-0.5B-VocalNet
export PROMPT_SPEECH="/home/tuwenming/Projects/URO-Bench/examples/VocalNet-test/omni_speech/infer/alloy.wav"                                 # 可选：自定义prompt音频路径，默认使用内置音频

# 目录配置（必需）
code_dir="/home/tuwenming/Projects/URO-Bench"                                             # URO-Bench代码目录，例如：/path/to/URO-Bench
log_dir="/home/tuwenming/Projects/URO-Bench/log/VocalNet-1B"                                              # 评估结果保存目录，例如：/path/to/URO-Bench-log/VocalNet-test
whisper_dir="/home/tuwenming/Models/openai/whisper-large-v3"                                          # whisper-large-v3模型路径，例如：/path/to/whisper-large-v3
uro_data_dir="/home/tuwenming/Projects/URO-Bench/URO-Bench-data"                                         # URO-Bench数据目录，例如：/path/to/URO-Bench-data

# conda环境配置（必需）
conda_dir="/home/tuwenming/anaconda3/etc/profile.d/conda.sh"                                            # conda路径，例如：/path/to/miniconda3/etc/profile.d/conda.sh
sdm_env_name="vocalnet"                                         # VocalNet环境名称，例如：vocalnet
uro_env_name=uro                                        # URO-Bench环境名称，通常为uro

# API密钥配置（评估阶段需要）
openai_api_key="sk-proj-1234567890"                                       # OpenAI API密钥（用于开放式问题评估）
gemini_api_key="sk-proj-1234567890"                                       # Gemini API密钥（用于某些评估任务）

# 选择要测试的数据集 - 单轮对话
datasets=(
    "AlpacaEval 199 open basic en"
    "CommonEval 200 open basic en"
    "WildchatEval 349 open basic en"
    "StoralEval 201 semi-open basic en"
    "Summary 118 semi-open basic en"
    "TruthfulEval 470 semi-open basic en"
    "GaokaoEval 303 qa basic en"
    "Gsm8kEval 582 qa basic en"
    "MLC 177 qa basic en"
    "Repeat 252 wer basic en"
    "AlpacaEval-zh 147 open basic zh"
    "Claude-zh 222 open basic zh"
    "Wildchat-zh 299 open basic zh"
    "SQuAD-zh 153 qa basic zh"
    "APE-zh 190 qa basic zh"
    "MLC-zh 145 qa basic zh"
    "OpenbookQA-zh 189 qa basic zh"
    "HSK5-zh 100 qa basic zh"
    "LCSTS-zh 119 semi-open basic zh"
    "Repeat-zh 127 wer basic zh"
    "CodeSwitching-en 70 semi-open pro en"
    "CodeSwitching-zh 70 semi-open pro zh"
    "GenEmotion-en 54 ge pro en"
    "GenEmotion-zh 43 ge pro zh"
    "GenStyle-en 44 gs pro en"
    "GenStyle-zh 39 gs pro zh"
    "MLCpro-en 91 qa pro en"
    "MLCpro-zh 64 qa pro zh"
    "Safety-en 24 sf pro en"
    "Safety-zh 20 sf pro zh"
    "SRT-en 43 srt pro en"
    "SRT-zh 21 srt pro zh"
    "UnderEmotion-en 137 ue pro en"
    "UnderEmotion-zh 79 ue pro zh"
    "Multilingual 1108 ml pro en"
    "ClothoEval-en 265 qa pro en"
    "MuChoEval-en 311 qa pro en"
)

# 选择要测试的数据集 - 多轮对话
multi_datasets=(
    "MtBenchEval-en 190 multi pro en"
    "SpeakerAware-en 55 sa pro en"
    "SpeakerAware-zh 49 sa pro zh"
)