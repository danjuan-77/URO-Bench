#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH


# code dir
model_name=glm-4-9b-chat-hf
code_dir=/data/ruiqi.yan/URO-Bench
ckpt_dir=/data/ruiqi.yan/models/${model_name}
log_dir=/data/ruiqi.yan/URO-Bench-log/${model_name}-test
whisper_dir=/data/ruiqi.yan/models/whisper-large-v3

# all the datasets
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
    "AlpacaEval-zh 200 open basic zh"
    "Claude-zh 273 open basic zh"
    "LCSTS-zh 229 semi-open basic zh"
    "MLC-zh 136 qa basic zh"
    "OpenbookQA-zh 257 qa basic zh"
    "Repeat-zh 210 wer basic zh"
    "CodeSwitching-en 70 semi-open pro en"
    "CodeSwitching-zh 70 semi-open pro zh"
    "GenStyle-en 44 gs pro en"
    "GenStyle-zh 39 gs pro zh"
    "MLCpro-en 91 qa pro en"
    "MLCpro-zh 64 qa pro zh"
    "UnderEmotion-en 137 ue pro en"
    "UnderEmotion-zh 79 ue pro zh"
    "Multilingual 1108 ml pro en"
)

# eval
for pair in "${datasets[@]}"
do
    # get dataset info
    dataset_name=$(echo "$pair" | cut -d' ' -f1)
    sample_number=$(echo "$pair" | cut -d' ' -f2)
    eval_mode=$(echo "$pair" | cut -d' ' -f3)
    level=$(echo "$pair" | cut -d' ' -f4)
    language=$(echo "$pair" | cut -d' ' -f5)
    dataset_path=/data/ruiqi.yan/URO-Bench-data/${level}/${dataset_name}/test.jsonl

    # output dir
    infer_output_dir=${log_dir}/eval/${level}/${dataset_name}
    eval_output_dir=$infer_output_dir/eval

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    # put your env name here, this env depends on the model you are testing
    conda activate yrq-glm
    # inference
    # -m debugpy --listen 5678 --wait-for-client
    python $code_dir/examples/llm-test/inference_for_eval.py \
        --dataset $dataset_path \
        --modality "audio" \
        --output_dir $infer_output_dir \
        --whisper_path $whisper_dir \
        --llm_path $ckpt_dir

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    conda activate yrq-uro               # put your env name here

    # assign scores
    if [[ ${eval_mode} == "open" ]]; then
        python $code_dir/mark.py \
        --mode $eval_mode \
        --question $infer_output_dir/question_text.jsonl \
        --answer $infer_output_dir/pred_text.jsonl \
        --output_dir $eval_output_dir \
        --dataset $dataset_name \
        --dataset_path $dataset_path \
        --language $language
    else
        python $code_dir/mark.py \
        --mode $eval_mode \
        --question $infer_output_dir/question_text.jsonl \
        --answer $infer_output_dir/pred_text.jsonl \
        --output_dir $eval_output_dir \
        --dataset $dataset_name \
        --dataset_path $dataset_path \
        --language $language \
        --reference $infer_output_dir/gt_text.jsonl
    fi

done

# multi-round eval
# multi-round datasets
manifest_format=jsonl
datasets=(
    "MtBenchEval-en 190 multi pro en"
    "SpeakerAware-en 55 sa pro en"
    "SpeakerAware-zh 49 sa pro zh"
)

# eval
for pair in "${datasets[@]}"
do
    # get dataset info
    dataset_name=$(echo "$pair" | cut -d' ' -f1)
    sample_number=$(echo "$pair" | cut -d' ' -f2)
    eval_mode=$(echo "$pair" | cut -d' ' -f3)
    level=$(echo "$pair" | cut -d' ' -f4)
    language=$(echo "$pair" | cut -d' ' -f5)
    dataset_path=/data/ruiqi.yan/URO-Bench-data/${level}/${dataset_name}/test.jsonl

    # output dir
    infer_output_dir=${log_dir}/eval/${level}/${dataset_name}
    eval_output_dir=$infer_output_dir/eval

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    # put your env name here, this env depends on the model you are testing
    conda activate yrq-glm
    # inference
    # -m debugpy --listen 5678 --wait-for-client
    python $code_dir/examples/llm-test/inference_multi.py \
        --dataset $dataset_path \
        --modality "audio" \
        --output_dir $infer_output_dir \
        --whisper_path $whisper_dir \
        --llm_path $ckpt_dir
    

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    conda activate yrq-uro               # put your env name here

    # assign scores
    python $code_dir/mark.py \
    --mode $eval_mode \
    --question $infer_output_dir/output_with_text.jsonl \
    --answer $infer_output_dir/output_with_text.jsonl \
    --output_dir $eval_output_dir \
    --dataset $dataset_name \
    --dataset_path $dataset_path \
    --language $language

done

# conclusion
python $code_dir/evaluate.py --eval_dir ${log_dir}/eval --non_asr