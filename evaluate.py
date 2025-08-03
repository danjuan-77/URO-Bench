from tqdm import tqdm
from argparse import ArgumentParser
import logging
import os
import jsonlines
import numpy as np
import matplotlib.pyplot as plt


# Sorry, the code is not concise enough
def conclude(dir, non_asr, name_format):
    datasets = [
        "Repeat result_repeat_wer.jsonl basic en understanding_basic_en",
        "Summary result_semi_open.jsonl basic en understanding_basic_en",
        "GaokaoEval result_qa.jsonl basic en understanding_basic_en",
        "StoralEval result_semi_open.jsonl basic en reasoning_basic_en",
        "TruthfulEval result_semi_open.jsonl basic en reasoning_basic_en",
        "Gsm8kEval result_qa.jsonl basic en reasoning_basic_en",
        "MLC result_qa.jsonl basic en reasoning_basic_en",
        "AlpacaEval result_open.jsonl basic en oral_basic_en",
        "CommonEval result_open.jsonl basic en oral_basic_en",
        "WildchatEval result_open.jsonl basic en oral_basic_en",
        "Repeat-zh result_repeat_cer.jsonl basic zh understanding_basic_zh",
        "LCSTS-zh result_semi_open.jsonl basic zh understanding_basic_zh",
        "HSK5-zh result_qa.jsonl basic zh understanding_basic_zh",
        "SQuAD-zh result_qa.jsonl basic zh reasoning_basic_zh",
        "APE-zh result_qa.jsonl basic zh reasoning_basic_zh",
        "MLC-zh result_qa.jsonl basic zh reasoning_basic_zh",
        "OpenbookQA-zh result_qa.jsonl basic zh reasoning_basic_zh",
        "AlpacaEval-zh result_open.jsonl basic zh oral_basic_zh",
        "Claude-zh result_open.jsonl basic zh oral_basic_zh",
        "Wildchat-zh result_open.jsonl basic zh oral_basic_zh",
        "UnderEmotion-en result_ue.jsonl pro en understanding_pro_en",
        "CodeSwitching-en result_semi_open.jsonl pro en understanding_pro_en",
        "Safety-en result_sf.jsonl pro en understanding_pro_en",
        "ClothoEval-en result_qa.jsonl pro en understanding_pro_en",
        "MuChoEval-en result_qa.jsonl pro en understanding_pro_en",
        "MLCpro-en result_qa.jsonl pro en reasoning_pro_en",
        "MtBenchEval-en result_multi.jsonl pro en reasoning_pro_en",
        "SpeakerAware-en result_sa.jsonl pro en reasoning_pro_en",
        "SRT-en result_srt.jsonl pro en oral_pro_en",
        "GenEmotion-en result_ge.jsonl pro en oral_pro_en",
        "GenStyle-en result_gs.jsonl pro en oral_pro_en",
        "Multilingual result_ml.jsonl pro en oral_pro_en",
        "UnderEmotion-zh result_ue.jsonl pro zh understanding_pro_zh",
        "CodeSwitching-zh result_semi_open.jsonl pro zh understanding_pro_zh",
        "Safety-zh result_sf.jsonl pro zh understanding_pro_zh",
        "MLCpro-zh result_qa.jsonl pro zh reasoning_pro_zh",
        "SpeakerAware-zh result_sa.jsonl pro zh reasoning_pro_zh",
        "SRT-zh result_srt.jsonl pro zh oral_pro_zh",
        "GenEmotion-zh result_ge.jsonl pro zh oral_pro_zh",
        "GenStyle-zh result_gs.jsonl pro zh oral_pro_zh",
    ]
    score_sum = 0
    result_num = 0
    record_num = {
        "understanding_basic_en": 0,
        "reasoning_basic_en": 0,
        "oral_basic_en": 0,
        "understanding_basic_zh": 0,
        "reasoning_basic_zh": 0,
        "oral_basic_zh": 0,
        "understanding_pro_en": 0,
        "reasoning_pro_en": 0,
        "oral_pro_en": 0,
        "understanding_pro_zh": 0,
        "reasoning_pro_zh": 0,
        "oral_pro_zh": 0,
        "understanding_basic": 0,
        "reasoning_basic": 0,
        "oral_basic": 0,
        "understanding_pro": 0,
        "reasoning_pro": 0,
        "oral_pro": 0,
        "understanding_en": 0,
        "reasoning_en": 0,
        "oral_en": 0,
        "understanding_zh": 0,
        "reasoning_zh": 0,
        "oral_zh": 0,
        "basic_en": 0,
        "basic_zh": 0,
        "pro_en": 0,
        "pro_zh": 0,
        "understanding": 0,
        "reasoning": 0,
        "oral": 0,
        "basic": 0,
        "pro": 0,
        "en": 0,
        "zh": 0,
    }
    record_sum = {
        "understanding_basic_en": 0,
        "reasoning_basic_en": 0,
        "oral_basic_en": 0,
        "understanding_basic_zh": 0,
        "reasoning_basic_zh": 0,
        "oral_basic_zh": 0,
        "understanding_pro_en": 0,
        "reasoning_pro_en": 0,
        "oral_pro_en": 0,
        "understanding_pro_zh": 0,
        "reasoning_pro_zh": 0,
        "oral_pro_zh": 0,
        "understanding_basic": 0,
        "reasoning_basic": 0,
        "oral_basic": 0,
        "understanding_pro": 0,
        "reasoning_pro": 0,
        "oral_pro": 0,
        "understanding_en": 0,
        "reasoning_en": 0,
        "oral_en": 0,
        "understanding_zh": 0,
        "reasoning_zh": 0,
        "oral_zh": 0,
        "basic_en": 0,
        "basic_zh": 0,
        "pro_en": 0,
        "pro_zh": 0,
        "understanding": 0,
        "reasoning": 0,
        "oral": 0,
        "basic": 0,
        "pro": 0,
        "en": 0,
        "zh": 0,
    }
    utmos_en_sum = 0
    utmos_en_num = 0
    wer_sum = 0
    wer_num = 0
    utmos_en_basic_sum = 0
    utmos_en_basic_num = 0
    wer_basic_sum = 0
    wer_basic_num = 0
    utmos_en_pro_sum = 0
    utmos_en_pro_num = 0
    wer_pro_sum = 0
    wer_pro_num = 0
    utmos_zh_sum = 0
    utmos_zh_num = 0
    cer_sum = 0
    cer_num = 0
    utmos_zh_basic_sum = 0
    utmos_zh_basic_num = 0
    cer_basic_sum = 0
    cer_basic_num = 0
    utmos_zh_pro_sum = 0
    utmos_zh_pro_num = 0
    cer_pro_sum = 0
    cer_pro_num = 0
    with jsonlines.open(os.path.join(dir, "evaluation.jsonl"), mode="w") as r:
        for dataset_info in datasets:
            print(dataset_info)
            dataset_name, result, level, language, category = dataset_info.split()
            if name_format is not None:
                if not non_asr:
                    result_dir = os.path.join(
                        dir, f"{level}/{name_format}{dataset_name}/eval_with_asr"
                    )
                else:
                    result_dir = os.path.join(
                        dir, f"{level}/{name_format}{dataset_name}/eval"
                    )
            else:
                if not non_asr:
                    result_dir = os.path.join(
                        dir, f"{level}/{dataset_name}/eval_with_asr"
                    )
                else:
                    result_dir = os.path.join(dir, f"{level}/{dataset_name}/eval")
            result_file = os.path.join(result_dir, result)
            utmos_file = os.path.join(result_dir, "result_utmos.jsonl")
            if language == "en":
                wer_file = os.path.join(result_dir, "result_wer.jsonl")
            elif language == "zh":
                wer_file = os.path.join(result_dir, "result_cer.jsonl")
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    num = 0
                    for item in jsonlines.Reader(f):
                        data = item
                        num += 1
                    if dataset_name == "Repeat" or dataset_name == "Repeat-zh":
                        score = (
                            data["ok_rate"] * (1 - data["final_WER_for_ok_case"]) * 100
                        )
                    else:
                        score = data["final_score"]
                record_num[category] += 1
                record_sum[category] += score
                if "understanding" in category:
                    record_num["understanding"] += 1
                    record_sum["understanding"] += score
                    if "basic" in category:
                        record_num["understanding_basic"] += 1
                        record_sum["understanding_basic"] += score
                        record_num["basic"] += 1
                        record_sum["basic"] += score
                    if "pro" in category:
                        record_num["understanding_pro"] += 1
                        record_sum["understanding_pro"] += score
                        record_num["pro"] += 1
                        record_sum["pro"] += score
                    if "en" in category:
                        record_num["understanding_en"] += 1
                        record_sum["understanding_en"] += score
                        record_num["en"] += 1
                        record_sum["en"] += score
                    if "zh" in category:
                        record_num["understanding_zh"] += 1
                        record_sum["understanding_zh"] += score
                        record_num["zh"] += 1
                        record_sum["zh"] += score
                    if "basic_en" in category:
                        record_num["basic_en"] += 1
                        record_sum["basic_en"] += score
                    if "pro_en" in category:
                        record_num["pro_en"] += 1
                        record_sum["pro_en"] += score
                    if "basic_zh" in category:
                        record_num["basic_zh"] += 1
                        record_sum["basic_zh"] += score
                    if "pro_zh" in category:
                        record_num["pro_zh"] += 1
                        record_sum["pro_zh"] += score
                if "reasoning" in category:
                    record_num["reasoning"] += 1
                    record_sum["reasoning"] += score
                    if "basic" in category:
                        record_num["reasoning_basic"] += 1
                        record_sum["reasoning_basic"] += score
                        record_num["basic"] += 1
                        record_sum["basic"] += score
                    if "pro" in category:
                        record_num["reasoning_pro"] += 1
                        record_sum["reasoning_pro"] += score
                        record_num["pro"] += 1
                        record_sum["pro"] += score
                    if "en" in category:
                        record_num["reasoning_en"] += 1
                        record_sum["reasoning_en"] += score
                        record_num["en"] += 1
                        record_sum["en"] += score
                    if "zh" in category:
                        record_num["reasoning_zh"] += 1
                        record_sum["reasoning_zh"] += score
                        record_num["zh"] += 1
                        record_sum["zh"] += score
                    if "basic_en" in category:
                        record_num["basic_en"] += 1
                        record_sum["basic_en"] += score
                    if "pro_en" in category:
                        record_num["pro_en"] += 1
                        record_sum["pro_en"] += score
                    if "basic_zh" in category:
                        record_num["basic_zh"] += 1
                        record_sum["basic_zh"] += score
                    if "pro_zh" in category:
                        record_num["pro_zh"] += 1
                        record_sum["pro_zh"] += score
                if "oral" in category:
                    record_num["oral"] += 1
                    record_sum["oral"] += score
                    if "basic" in category:
                        record_num["oral_basic"] += 1
                        record_sum["oral_basic"] += score
                        record_num["basic"] += 1
                        record_sum["basic"] += score
                    if "pro" in category:
                        record_num["oral_pro"] += 1
                        record_sum["oral_pro"] += score
                        record_num["pro"] += 1
                        record_sum["pro"] += score
                    if "en" in category:
                        record_num["oral_en"] += 1
                        record_sum["oral_en"] += score
                        record_num["en"] += 1
                        record_sum["en"] += score
                    if "zh" in category:
                        record_num["oral_zh"] += 1
                        record_sum["oral_zh"] += score
                        record_num["zh"] += 1
                        record_sum["zh"] += score
                    if "basic_en" in category:
                        record_num["basic_en"] += 1
                        record_sum["basic_en"] += score
                    if "pro_en" in category:
                        record_num["pro_en"] += 1
                        record_sum["pro_en"] += score
                    if "basic_zh" in category:
                        record_num["basic_zh"] += 1
                        record_sum["basic_zh"] += score
                    if "pro_zh" in category:
                        record_num["pro_zh"] += 1
                        record_sum["pro_zh"] += score
                result_num += 1
                score_sum += score
                r.write({"samples": num - 1, f"score on {dataset_name}": score})
            if language == "en":
                if os.path.exists(utmos_file):
                    with open(utmos_file, "r") as f:
                        for item in jsonlines.Reader(f):
                            data = item
                        utmos_en_num += 1
                        utmos_en_sum += data["final_UTMOS"]
                        if "basic" in category:
                            utmos_en_basic_num += 1
                            utmos_en_basic_sum += data["final_UTMOS"]
                        if "pro" in category:
                            utmos_en_pro_num += 1
                            utmos_en_pro_sum += data["final_UTMOS"]
                if os.path.exists(wer_file):
                    with open(wer_file, "r") as f:
                        for item in jsonlines.Reader(f):
                            data = item
                        wer_num += 1
                        wer_sum += data["final_WER"]
                        if "basic" in category:
                            wer_basic_num += 1
                            wer_basic_sum += data["final_WER"]
                        if "pro" in category:
                            wer_pro_num += 1
                            wer_pro_sum += data["final_WER"]
            elif language == "zh":
                if os.path.exists(utmos_file):
                    with open(utmos_file, "r") as f:
                        for item in jsonlines.Reader(f):
                            data = item
                        utmos_zh_num += 1
                        utmos_zh_sum += data["final_UTMOS"]
                        if "basic" in category:
                            utmos_zh_basic_num += 1
                            utmos_zh_basic_sum += data["final_UTMOS"]
                        if "pro" in category:
                            utmos_zh_pro_num += 1
                            utmos_zh_pro_sum += data["final_UTMOS"]
                if os.path.exists(wer_file):
                    with open(wer_file, "r") as f:
                        for item in jsonlines.Reader(f):
                            data = item
                        cer_num += 1
                        cer_sum += data["final_WER"]
                        if "basic" in category:
                            cer_basic_num += 1
                            cer_basic_sum += data["final_WER"]
                        if "pro" in category:
                            cer_pro_num += 1
                            cer_pro_sum += data["final_WER"]

        for category_name, category_num in record_num.items():
            if category_num != 0:
                r.write(
                    {
                        f"final score on {category_name}": record_sum[category_name]
                        / category_num
                    }
                )
        if utmos_en_basic_num != 0:
            r.write({"final_UTMOS_en_basic": utmos_en_basic_sum / utmos_en_basic_num})
        if utmos_zh_basic_num != 0:
            r.write({"final_UTMOS_zh_basic": utmos_zh_basic_sum / utmos_zh_basic_num})
        if utmos_en_pro_num != 0:
            r.write({"final_UTMOS_en_pro": utmos_en_pro_sum / utmos_en_pro_num})
        if utmos_zh_pro_num != 0:
            r.write({"final_UTMOS_zh_pro": utmos_zh_pro_sum / utmos_zh_pro_num})
        if wer_basic_num != 0:
            r.write({"final_WER_basic": wer_basic_sum / wer_basic_num})
        if cer_basic_num != 0:
            r.write({"final_CER_basic": cer_basic_sum / cer_basic_num})
        if wer_pro_num != 0:
            r.write({"final_WER_pro": wer_pro_sum / wer_pro_num})
        if cer_pro_num != 0:
            r.write({"final_CER_pro": cer_pro_sum / cer_pro_num})
        if result_num != 0:
            r.write({"final_score": score_sum / result_num})
        if utmos_en_num != 0:
            r.write({"final_UTMOS_en": utmos_en_sum / utmos_en_num})
        if utmos_zh_num != 0:
            r.write({"final_UTMOS_zh": utmos_zh_sum / utmos_zh_num})
        if wer_num != 0:
            r.write({"final_WER": wer_sum / wer_num})
        if cer_num != 0:
            r.write({"final_CER": cer_sum / cer_num})
    radar_data_en = []
    radar_data_zh = []
    flag_en = False
    flag_zh = False
    capability_en = [
        "understanding_basic_en",
        "reasoning_basic_en",
        "oral_basic_en",
        "understanding_pro_en",
        "reasoning_pro_en",
        "oral_pro_en",
    ]
    capability_zh = [
        "understanding_basic_zh",
        "reasoning_basic_zh",
        "oral_basic_zh",
        "understanding_pro_zh",
        "reasoning_pro_zh",
        "oral_pro_zh",
    ]
    for s in capability_en:
        if record_num[s] != 0:
            radar_data_en.append(record_sum[s] / record_num[s])
            flag_en = True
        else:
            radar_data_en.append(0.00)
    for s in capability_zh:
        if record_num[s] != 0:
            radar_data_zh.append(record_sum[s] / record_num[s])
            flag_zh = True
        else:
            radar_data_zh.append(0.00)

    labels = [
        "Understanding Basic",
        "Reasoning Basic",
        "Oral Basic",
        "Understanding Pro",
        "Reasoning Pro",
        "Oral Pro",
    ]

    if flag_en:
        angles1 = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        radar_data_en += radar_data_en[:1]
        angles1 += angles1[:1]
        fig1, ax1 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax1.set_theta_offset(np.pi * 2 / 3)
        ax1.set_facecolor("#cfd3d4")
        ax1.grid(True, color="white", linestyle="-", linewidth=0.8)
        ax1.plot(
            angles1,
            radar_data_en,
            linewidth=2,
            linestyle="solid",
            label="proficiency_en",
        )
        ax1.set_yticklabels([])
        ax1.set_xticks(angles1[:-1])
        ax1.set_xticklabels(labels)
        ax1.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        fig1.savefig(os.path.join(dir, "proficiency_en.png"))

    if flag_zh:
        angles2 = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        radar_data_zh += radar_data_zh[:1]
        angles2 += angles2[:1]
        fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax2.set_theta_offset(np.pi * 2 / 3)
        ax2.set_facecolor("#cfd3d4")
        ax2.grid(True, color="white", linestyle="-", linewidth=0.8)
        ax2.plot(
            angles2,
            radar_data_zh,
            linewidth=2,
            linestyle="solid",
            label="proficiency_zh",
        )
        ax2.set_yticklabels([])
        ax2.set_xticks(angles2[:-1])
        ax2.set_xticklabels(labels)
        ax2.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        fig2.savefig(os.path.join(dir, "proficiency_zh.png"))


def main():
    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--non_asr", action="store_true")
    parser.add_argument("--name_format", required=False)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # evaluate
    dir = args.eval_dir
    non_asr = args.non_asr
    name_format = args.name_format
    conclude(dir, non_asr, name_format)
    logging.info(f"result saved to {os.path.join(dir, 'evaluation.jsonl')}")


if __name__ == "__main__":
    main()
