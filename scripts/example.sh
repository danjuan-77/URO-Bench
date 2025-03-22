#!/bin/bash
config_path=$1
source ${config_path}

# eval
for pair in "${datasets[@]}"
do
    # get dataset info
    dataset_name=$(echo "$pair" | cut -d' ' -f1)
    sample_number=$(echo "$pair" | cut -d' ' -f2)
    eval_mode=$(echo "$pair" | cut -d' ' -f3)
    level=$(echo "$pair" | cut -d' ' -f4)
    language=$(echo "$pair" | cut -d' ' -f5)
    dataset_path=${uro_data_dir}/${level}/${dataset_name}/test.jsonl

    # output dir
    infer_output_dir=${log_dir}/eval/${level}/${dataset_name}
    eval_output_dir=$infer_output_dir/eval_with_asr

# ============================Modify Me====================================================

    source ${conda_dir}
    conda activate ${sdm_env_name}

    # inference, please modify this part according to your "inference_for_eval.py"
    cd $code_dir/examples/${model_name}-test
    python $code_dir/examples/${model_name}-test/inference_for_eval.py \
        --output_dir $infer_output_dir \
        --dataset $dataset_path

# ============================Modify Me====================================================

    source ${conda_dir}
    conda activate ${uro_env_name}
    # asr
    python $code_dir/asr_for_eval.py \
        --input_dir $infer_output_dir/audio \
        --model_dir $whisper_dir \
        --output_dir $infer_output_dir \
        --language $language \
        --number $sample_number

    # assign scores
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

done

# multi-round eval
for pair in "${multi_datasets[@]}"
do
    # get dataset info
    dataset_name=$(echo "$pair" | cut -d' ' -f1)
    sample_number=$(echo "$pair" | cut -d' ' -f2)
    eval_mode=$(echo "$pair" | cut -d' ' -f3)
    level=$(echo "$pair" | cut -d' ' -f4)
    language=$(echo "$pair" | cut -d' ' -f5)
    dataset_path=${uro_data_dir}/${level}/${dataset_name}/test.jsonl

    # output dir
    infer_output_dir=${log_dir}/eval/${level}/${dataset_name}
    eval_output_dir=$infer_output_dir/eval_with_asr

# ============================Modify Me====================================================

    source ${conda_dir}
    conda activate ${sdm_env_name}

    # inference, please modify this part according to your "inference_multi.py"
    cd $code_dir/examples/${model_name}-test
    python $code_dir/examples/${model_name}-test/inference_multi.py \
        --dataset $dataset_path \
        --output_dir $infer_output_dir \

# ============================Modify Me====================================================

    source ${conda_dir}
    conda activate ${uro_env_name}
    # asr
    python $code_dir/asr_for_eval.py \
        --input_dir $infer_output_dir \
        --model_dir $whisper_dir \
        --output_dir $infer_output_dir \
        --language $language \
        --number $sample_number \
        --dataset $dataset_path \
        --multi

    # assign scores
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

done

# conclusion
python $code_dir/evaluate.py --eval_dir ${log_dir}/eval