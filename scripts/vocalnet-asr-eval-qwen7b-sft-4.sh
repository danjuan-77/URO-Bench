#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT=https://hf-mirror.com

# VocalNet模型路径配置
export VOCALNET_MODEL="/mnt/buffer/tuwenming/checkpoints/VocalNet/qwen25-7b-instruct-s2s-mtp-ultravoice100k-clean-all-sft-llm-and-decoder/checkpoint-4611"        # VocalNet模型路径，需要根据实际情况填写
export COSYVOICE_MODEL="/share/nlp/tuwenming/models/CosyVoice/CosyVoice2-0.5B-old"       # CosyVoice2-0.5B模型路径，需要根据实际情况填写
export PROMPT_SPEECH="/share/nlp/tuwenming/projects/URO-Bench/examples/VocalNet-test/omni_speech/infer/alloy.wav"

# code dir
model_name=VocalNet
code_dir="/share/nlp/tuwenming/projects/URO-Bench"                           # URO-Bench代码目录，需要根据实际情况填写
log_dir="/share/nlp/kangyipeng/infer_results/URO-Bench/VocalNet-Qwen25-7B-UltraVoice100k-Clean-Steps4611-SFT"     # 评估结果保存目录，需要根据实际情况填写
whisper_dir="/share/nlp/tuwenming/models/openai/whisper-large-v3"                 # whisper模型路径，需要根据实际情况填写
uro_data_dir="/share/nlp/tuwenming/projects/URO-Bench/URO-Bench-data"                  # URO-Bench数据目录，需要根据实际情况填写

# conda环境配置
conda_dir="/home/tuwenming/anaconda3/etc/profile.d/conda.sh"  # conda路径，需要根据实际情况填写
vocalnet_env_name=vocalnet                             # VocalNet环境名称，需要根据实际情况填写
uro_env_name=uro                                       # URO-Bench环境名称

# OpenAI和Gemini API密钥（用于评估）
openai_api_key="sk-proj-1234567890"                                      # OpenAI API密钥，需要根据实际情况填写
gemini_api_key="sk-proj-1234567890"                                      # Gemini API密钥，需要根据实际情况填写

# 所有单轮对话数据集
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
    # # "AlpacaEval-zh 147 open basic zh"
    # # "Claude-zh 222 open basic zh"
    # # "Wildchat-zh 299 open basic zh"
    # # "SQuAD-zh 153 qa basic zh"
    # # "APE-zh 190 qa basic zh"
    # # "MLC-zh 145 qa basic zh"
    # # "OpenbookQA-zh 189 qa basic zh"
    # # "HSK5-zh 100 qa basic zh"
    # # "LCSTS-zh 119 semi-open basic zh"
    # # "Repeat-zh 127 wer basic zh"
    "CodeSwitching-en 70 semi-open pro en"
    # # "CodeSwitching-zh 70 semi-open pro zh"
    "GenEmotion-en 54 ge pro en"
    # # "GenEmotion-zh 43 ge pro zh"
    "GenStyle-en 44 gs pro en"
    # # "GenStyle-zh 39 gs pro zh"
    "MLCpro-en 91 qa pro en"
    # # "MLCpro-zh 64 qa pro zh"
    "Safety-en 24 sf pro en"
    # # "Safety-zh 20 sf pro zh"
    "SRT-en 43 srt pro en"
    # # "SRT-zh 21 srt pro zh"
    "UnderEmotion-en 137 ue pro en"
    # # "UnderEmotion-zh 79 ue pro zh"
    # "Multilingual 1108 ml pro en"
    # "ClothoEval-en 265 qa pro en"
    # "MuChoEval-en 311 qa pro en"
)

# 单轮对话评估
for pair in "${datasets[@]}"
do
    # 获取数据集信息
    dataset_name=$(echo "$pair" | cut -d' ' -f1)
    sample_number=$(echo "$pair" | cut -d' ' -f2)
    eval_mode=$(echo "$pair" | cut -d' ' -f3)
    level=$(echo "$pair" | cut -d' ' -f4)
    language=$(echo "$pair" | cut -d' ' -f5)
    dataset_path=${uro_data_dir}/${level}/${dataset_name}/test.jsonl

    # 输出目录
    infer_output_dir=${log_dir}/eval/${level}/${dataset_name}
    eval_output_dir=$infer_output_dir/eval_with_asr

    echo "开始处理数据集: ${dataset_name}"
    
    # 激活VocalNet环境并进行推理
    source ${conda_dir}
    conda activate ${vocalnet_env_name}

    # 推理
    cd $code_dir/examples/${model_name}-test
    python $code_dir/examples/${model_name}-test/inference_for_eval.py \
        --dataset $dataset_path \
        --output_dir $infer_output_dir

    # # 激活URO-Bench环境进行ASR和评分
    # source ${conda_dir}
    # conda activate ${uro_env_name}
    
    # # ASR转录
    # python $code_dir/asr_for_eval.py \
    #     --input_dir $infer_output_dir/audio \
    #     --model_dir $whisper_dir \
    #     --output_dir $infer_output_dir \
    #     --language $language \
    #     --number $sample_number

    # # 评分
    # if [[ ${eval_mode} == "open" ]]; then
    #     python $code_dir/mark.py \
    #     --mode $eval_mode \
    #     --question $infer_output_dir/question_text.jsonl \
    #     --answer $infer_output_dir/asr_text.jsonl \
    #     --answer_text $infer_output_dir/pred_text.jsonl \
    #     --output_dir $eval_output_dir \
    #     --dataset $dataset_name \
    #     --dataset_path $dataset_path \
    #     --language $language \
    #     --audio_dir $infer_output_dir/audio \
    #     --openai_api_key $openai_api_key
    # else
    #     python $code_dir/mark.py \
    #     --mode $eval_mode \
    #     --question $infer_output_dir/question_text.jsonl \
    #     --answer $infer_output_dir/asr_text.jsonl \
    #     --answer_text $infer_output_dir/pred_text.jsonl \
    #     --output_dir $eval_output_dir \
    #     --dataset $dataset_name \
    #     --dataset_path $dataset_path \
    #     --language $language \
    #     --audio_dir $infer_output_dir/audio \
    #     --reference $infer_output_dir/gt_text.jsonl \
    #     --openai_api_key $openai_api_key \
    #     --gemini_api_key $gemini_api_key
    # fi

    echo "完成数据集: ${dataset_name}"
done

# 多轮对话数据集
multi_datasets=(
    # "MtBenchEval-en 190 multi pro en"
    # "SpeakerAware-en 55 sa pro en"
    # "SpeakerAware-zh 49 sa pro zh"
)

# 多轮对话评估
for pair in "${multi_datasets[@]}"
do
    # 获取数据集信息
    dataset_name=$(echo "$pair" | cut -d' ' -f1)
    sample_number=$(echo "$pair" | cut -d' ' -f2)
    eval_mode=$(echo "$pair" | cut -d' ' -f3)
    level=$(echo "$pair" | cut -d' ' -f4)
    language=$(echo "$pair" | cut -d' ' -f5)
    dataset_path=${uro_data_dir}/${level}/${dataset_name}/test.jsonl

    # 输出目录
    infer_output_dir=${log_dir}/eval/${level}/${dataset_name}
    eval_output_dir=$infer_output_dir/eval_with_asr

    echo "开始处理多轮对话数据集: ${dataset_name}"
    
    # 激活VocalNet环境并进行推理
    source ${conda_dir}
    conda activate ${vocalnet_env_name}

    # 多轮对话推理
    cd $code_dir/examples/${model_name}-test
    python $code_dir/examples/${model_name}-test/inference_multi.py \
        --dataset $dataset_path \
        --output_dir $infer_output_dir

    # # 激活URO-Bench环境进行ASR和评分
    # source ${conda_dir}
    # conda activate ${uro_env_name}
    
    # # ASR转录
    # python $code_dir/asr_for_eval.py \
    #     --input_dir $infer_output_dir \
    #     --model_dir $whisper_dir \
    #     --output_dir $infer_output_dir \
    #     --language $language \
    #     --number $sample_number \
    #     --dataset $dataset_path \
    #     --multi

    # # 评分
    # python $code_dir/mark.py \
    # --mode $eval_mode \
    # --question $infer_output_dir/asr_text.jsonl \
    # --answer $infer_output_dir/asr_text.jsonl \
    # --answer_text $infer_output_dir/output_with_text.jsonl \
    # --output_dir $eval_output_dir \
    # --dataset $dataset_name \
    # --dataset_path $dataset_path \
    # --language $language \
    # --audio_dir $infer_output_dir \
    # --openai_api_key $openai_api_key

    echo "完成多轮对话数据集: ${dataset_name}"
done

# # 总结评估结果
# echo "开始生成评估总结..."
# python $code_dir/evaluate.py --eval_dir ${log_dir}/eval
# echo "评估完成！"


# nohup bash ./scripts/vocalnet-asr-eval-qwen7b-sft-4.sh > /share/nlp/kangyipeng/logs/VocalNet-Qwen25-7B-SFT-eval-$(date +%Y%m%d%H%M%S).log 2>&1 &
