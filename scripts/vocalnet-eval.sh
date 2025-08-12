#!/bin/bash

# VocalNet URO-Bench 评估脚本
# 使用方法：bash scripts/vocalnet-eval.sh scripts/vocalnet-config.sh

config_path=$1

if [ -z "$config_path" ]; then
    echo "错误：请提供配置文件路径"
    echo "使用方法：bash scripts/vocalnet-eval.sh scripts/vocalnet-config.sh"
    exit 1
fi

if [ ! -f "$config_path" ]; then
    echo "错误：配置文件不存在：$config_path"
    exit 1
fi

# 加载配置文件
source ${config_path}

echo "开始VocalNet模型评估..."
echo "配置文件：$config_path"
echo "模型名称：$model_name"

# 检查必需的环境变量
if [ -z "$VOCALNET_MODEL" ]; then
    echo "错误：请在配置文件中设置 VOCALNET_MODEL"
    exit 1
fi

if [ -z "$COSYVOICE_MODEL" ]; then
    echo "错误：请在配置文件中设置 COSYVOICE_MODEL"
    exit 1
fi

if [ -z "$code_dir" ]; then
    echo "错误：请在配置文件中设置 code_dir"
    exit 1
fi

if [ -z "$uro_data_dir" ]; then
    echo "错误：请在配置文件中设置 uro_data_dir"
    exit 1
fi

echo "VOCALNET_MODEL: $VOCALNET_MODEL"
echo "COSYVOICE_MODEL: $COSYVOICE_MODEL"

# 单轮对话评估
echo "开始单轮对话评估..."
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

    echo "======================================"
    echo "处理数据集: ${dataset_name}"
    echo "样本数量: ${sample_number}"
    echo "评估模式: ${eval_mode}"
    echo "语言: ${language}"
    echo "======================================"
    
    # 激活VocalNet环境并进行推理
    source ${conda_dir}
    conda activate ${sdm_env_name}

    # 推理
    cd $code_dir/examples/${model_name}-test
    python $code_dir/examples/${model_name}-test/inference_for_eval.py \
        --dataset $dataset_path \
        --output_dir $infer_output_dir

    if [ $? -ne 0 ]; then
        echo "推理失败：${dataset_name}"
        continue
    fi

    # 激活URO-Bench环境进行ASR和评分
    source ${conda_dir}
    conda activate ${uro_env_name}
    
    # ASR转录
    python $code_dir/asr_for_eval.py \
        --input_dir $infer_output_dir/audio \
        --model_dir $whisper_dir \
        --output_dir $infer_output_dir \
        --language $language \
        --number $sample_number

    if [ $? -ne 0 ]; then
        echo "ASR转录失败：${dataset_name}"
        continue
    fi

    # 评分
    if [[ ${eval_mode} == "open" ]]; then
        python $code_dir/mark.py \
        --mode $eval_mode \
        --question $infer_output_dir/question_text.jsonl \
        --answer $infer_output_dir/asr_text.jsonl \
        --answer_text $infer_output_dir/pred_text.jsonl \
        --output_dir $eval_output_dir \
        --dataset $dataset_name \
        --dataset_path $dataset_path \
        --language $language \
        --audio_dir $infer_output_dir/audio \
        --openai_api_key $openai_api_key
    else
        python $code_dir/mark.py \
        --mode $eval_mode \
        --question $infer_output_dir/question_text.jsonl \
        --answer $infer_output_dir/asr_text.jsonl \
        --answer_text $infer_output_dir/pred_text.jsonl \
        --output_dir $eval_output_dir \
        --dataset $dataset_name \
        --dataset_path $dataset_path \
        --language $language \
        --audio_dir $infer_output_dir/audio \
        --reference $infer_output_dir/gt_text.jsonl \
        --openai_api_key $openai_api_key \
        --gemini_api_key $gemini_api_key
    fi

    if [ $? -eq 0 ]; then
        echo "完成数据集: ${dataset_name}"
    else
        echo "评分失败：${dataset_name}"
    fi
done

# 多轮对话评估
echo "开始多轮对话评估..."
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

    echo "======================================"
    echo "处理多轮对话数据集: ${dataset_name}"
    echo "样本数量: ${sample_number}"
    echo "语言: ${language}"
    echo "======================================"
    
    # 激活VocalNet环境并进行推理
    source ${conda_dir}
    conda activate ${sdm_env_name}

    # 多轮对话推理
    cd $code_dir/examples/${model_name}-test
    python $code_dir/examples/${model_name}-test/inference_multi.py \
        --dataset $dataset_path \
        --output_dir $infer_output_dir

    if [ $? -ne 0 ]; then
        echo "多轮对话推理失败：${dataset_name}"
        continue
    fi

    # 激活URO-Bench环境进行ASR和评分
    source ${conda_dir}
    conda activate ${uro_env_name}
    
    # ASR转录
    python $code_dir/asr_for_eval.py \
        --input_dir $infer_output_dir \
        --model_dir $whisper_dir \
        --output_dir $infer_output_dir \
        --language $language \
        --number $sample_number \
        --dataset $dataset_path \
        --multi

    if [ $? -ne 0 ]; then
        echo "多轮对话ASR转录失败：${dataset_name}"
        continue
    fi

    # 评分
    python $code_dir/mark.py \
    --mode $eval_mode \
    --question $infer_output_dir/asr_text.jsonl \
    --answer $infer_output_dir/asr_text.jsonl \
    --answer_text $infer_output_dir/output_with_text.jsonl \
    --output_dir $eval_output_dir \
    --dataset $dataset_name \
    --dataset_path $dataset_path \
    --language $language \
    --audio_dir $infer_output_dir \
    --openai_api_key $openai_api_key

    if [ $? -eq 0 ]; then
        echo "完成多轮对话数据集: ${dataset_name}"
    else
        echo "多轮对话评分失败：${dataset_name}"
    fi
done

# 总结评估结果
echo "======================================"
echo "开始生成评估总结..."
echo "======================================"
python $code_dir/evaluate.py --eval_dir ${log_dir}/eval

if [ $? -eq 0 ]; then
    echo "======================================"
    echo "VocalNet模型评估完成！"
    echo "结果保存在：${log_dir}/eval"
    echo "======================================"
else
    echo "评估总结生成失败！"
fi