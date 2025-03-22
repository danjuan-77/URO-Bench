from tqdm import tqdm
from google import genai
from argparse import ArgumentParser
import logging
import os
import jsonlines


# eval with gemini
def eval(args):
    # set your api key
    client = genai.Client(api_key=args.gemini_api_key)
    logging.info("<------start Gemini eval------>")

    if args.mode == "srt":
        output_file = os.path.join(args.output_dir, "result_srt.jsonl")
        sum_score = 0

        template = """
        I need your help to evaluate the performance of several models in a speech interaction scenario where the model is required to perform tasks such as singing, reciting, or reading tongue twisters. 
        The models will receive a user input and generate an audio response.
        Your task is to rate the model’s performance based on the provided user input transcription [Instruction] and the model’s audio output.

        Please evaluate the response on a scale of 1 to 5, focusing on the quality, clarity, and effectiveness of the audio output:
        1 point: The audio response is largely irrelevant or incorrect. The model fails to perform the requested task (singing, reciting, or reading) properly, or the audio is unclear, garbled, or hard to understand.
        2 points: The audio response somewhat matches the task, but with noticeable issues. The performance may be off-key or unclear, and the model may not fully follow the requested task (e.g., missing lyrics in a song or stumbling over words in a tongue twister).
        3 points: The audio response is generally clear and relevant, but it may lack fluency or accuracy in certain parts. The model performs the task reasonably well, but there may be slight mistakes or a lack of engagement in the delivery.
        4 points: The audio response is clear, accurate, and demonstrates a strong understanding of the task. The model performs the task effectively, but there may be minor inconsistencies or slight imperfections in delivery (e.g., minor timing or pitch issues in singing).
        5 points: The audio response is flawless, demonstrating full mastery of the task. The model performs the task with high clarity, accuracy, and engagement, delivering a high-quality performance that aligns perfectly with the user’s input and intent.

        Below is the transcription of user’s instruction:
        ### [Instruction]
        {question}

        After evaluating, please output the score only without anything else.
        You don’t need to provide any explanations.
        """

        with open(args.question, "r") as f:
            length = sum([1 for _ in f])
        with open(args.question, "r") as qt, jsonlines.open(
            output_file, mode="w"
        ) as ot:
            for i, question in tqdm(
                enumerate(jsonlines.Reader(qt)),
                total=length,
            ):
                item = {"question": question[str(i).zfill(4)]}
                file_path = os.path.join(args.audio_dir, str(i).zfill(4) + ".wav")
                myfile = client.files.upload(file=file_path)
                prompt = template.replace("{question}", item["question"])
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        prompt,
                        myfile,
                    ],
                )
                score = response.text
                try:
                    score = int(score)
                except:
                    score = 0
                item["score"] = score
                ot.write(item)
                sum_score += int(item["score"])
            ot.write({"final_score": sum_score * 20 / length})
        # save results
        logging.info(f"saving result to {output_file}")
