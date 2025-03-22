from tqdm import tqdm
import multiprocessing
from openai import OpenAI
from argparse import ArgumentParser
import logging
import os
import jsonlines
import metrics.mark_gpt as mark_gpt
import metrics.mark_wer as mark_wer
import metrics.mark_utmos as mark_utmos
import metrics.mark_emotion as mark_emotion
import metrics.mark_gemini as mark_gemini


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "qa",
            "open",
            "semi-open",
            "wer",
            "multi",
            "ge",
            "gs",
            "sf",
            "srt",
            "ue",
            "ml",
            "sa",
        ],
        help="mode of scoring criteria",
    )
    parser.add_argument(
        "--question", type=str, required=True, help="file of question text"
    )
    parser.add_argument(
        "--answer", type=str, required=True, help="file of response transcript"
    )
    parser.add_argument(
        "--answer_text",
        type=str,
        required=False,
        help="reference response text to calculate wer",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=False,
        help="output audio dir for measuring utmos",
    )
    parser.add_argument(
        "--reference", type=str, required=False, help="file of reference answer"
    )
    parser.add_argument(
        "--openai_api_key", type=str, required=False, help="openai api key"
    )
    parser.add_argument(
        "--gemini_api_key", type=str, required=False, help="gemini api key"
    )
    # parser.add_argument("--answer_contrast", type=str, required=False)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # evaluate data
    logging.info("<========start to evaluate data========>")

    # eval for output text
    if args.mode == "wer":
        mark_wer.eval_repeat(args)
    elif args.mode == "ge":
        mark_emotion.eval_ge(args)
    elif args.mode == "srt":
        mark_gemini.eval(args)
    else:
        mark_gpt.eval(args)

    # eval for output audio
    if args.audio_dir is not None:
        mark_utmos.eval(args)
    if args.answer_text is not None and args.dataset != "Multilingual":
        mark_wer.eval_wav(args)


if __name__ == "__main__":
    main()
