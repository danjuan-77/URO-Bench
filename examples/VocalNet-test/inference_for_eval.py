#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import logging
import jsonlines
import shutil
import torch
import torchaudio
from tqdm import tqdm

# Add the VocalNet-test directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import VocalNet model
from omni_speech.infer.vocalnet import VocalNetModel

# Global model instance
vocalnet_model = None

def load_sdm():
    """
    Load VocalNet SDM (Spoken Dialogue Model)
    """
    global vocalnet_model
    
    # Set model paths - you need to modify these paths according to your setup
    VOCALNET_MODEL_PATH = os.getenv("VOCALNET_MODEL")
    COSYVOICE_MODEL_PATH = os.getenv("COSYVOICE_MODEL")   
    
    if not VOCALNET_MODEL_PATH:
        raise ValueError("Please set VOCALNET_MODEL_PATH environment variable")
    if not COSYVOICE_MODEL_PATH:
        raise ValueError("Please set COSYVOICE_MODEL_PATH environment variable")
    
    logging.info("Initializing VocalNet model...")
    
    # Initialize VocalNet with speech-to-speech capability
    vocalnet_model = VocalNetModel(
        model_name_or_path=VOCALNET_MODEL_PATH,
        vocoder_path=COSYVOICE_MODEL_PATH,
        s2s=True,  # Enable speech-to-speech
    )
    
    # Initialize the model
    vocalnet_model.__initilize__()
    
    logging.info("VocalNet model loaded successfully!")
    return vocalnet_model


def respond(input_audio, output_path):
    """
    Generate response for input audio
    
    Args:
        input_audio: Path to input audio file
        output_path: Path where output audio should be saved
        
    Returns:
        str: Text response from the model
    """
    global vocalnet_model
    
    if vocalnet_model is None:
        raise RuntimeError("Model not loaded. Please call load_sdm() first.")
    
    # Set output directory for audio
    output_dir = os.path.dirname(output_path)
    vocalnet_model.set_audio_dir(output_dir)
    
    # Prepare input message in the format expected by VocalNet
    messages = [{
        'role': 'user', 
        'content': '<speech>', 
        'path': input_audio
    }]
    
    try:
        # Call VocalNet model
        result = vocalnet_model(messages)
        
        # Extract text response
        text_response = result.get('text', '').strip()
        
        # Handle audio output
        if 'audio' in result and result['audio']:
            generated_audio_path = result['audio']
            # Move/copy the generated audio to the expected output path
            if generated_audio_path != output_path:
                shutil.move(generated_audio_path, output_path)
        else:
            # If no audio is generated, create a silent audio file
            logging.warning(f"No audio generated for {input_audio}, creating silent audio")
            silent_audio = torch.zeros(1, 16000)  # 1 second of silence at 16kHz
            torchaudio.save(output_path, silent_audio, 16000)
        
        return text_response
        
    except Exception as e:
        logging.error(f"Error processing {input_audio}: {str(e)}")
        # Return empty response and create silent audio on error
        silent_audio = torch.zeros(1, 16000)
        torchaudio.save(output_path, silent_audio, 16000)
        return ""


def main():
    parser = argparse.ArgumentParser(description="VocalNet inference for URO-Bench evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset jsonl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create output directories
    output_dir = args.output_dir
    output_audio_dir = os.path.join(args.output_dir, "audio")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir, exist_ok=True)
    
    # Define output files
    pred_text = os.path.join(output_dir, "pred_text.jsonl")
    question_text = os.path.join(output_dir, "question_text.jsonl")
    gt_text = os.path.join(output_dir, "gt_text.jsonl")

    logging.info("<========Loading VocalNet model========>")
    load_sdm()

    logging.info("<========Starting inference========>")
    
    # Count total samples for progress bar
    with open(args.dataset, "r") as f:
        total_samples = sum(1 for _ in jsonlines.Reader(f))
    
    with open(args.dataset, "r") as f, \
         jsonlines.open(pred_text, mode="w") as pt, \
         jsonlines.open(question_text, mode="w") as qt, \
         jsonlines.open(gt_text, mode="w") as gt:
        
        for step, item in enumerate(tqdm(jsonlines.Reader(f), total=total_samples, desc="Processing")):
            # Get input and target information
            input_path = os.path.join(os.path.dirname(args.dataset), item["source_wav"])
            input_text = item["source_text"]
            
            if "target_text" in item:
                target_text = item["target_text"]
            else:
                target_text = item["source_text"]
            
            # Define output audio path
            output_path = os.path.join(output_audio_dir, f"{step:04d}.wav")
            
            # Generate response
            try:
                response = respond(input_path, output_path)
                logging.info(f"Sample {step:04d} - Input: {input_text[:100]}...")
                logging.info(f"Sample {step:04d} - Output: {response[:100]}...")
                logging.info(f"Sample {step:04d} - Audio saved to: {output_path}")
            except Exception as e:
                logging.error(f"Error processing sample {step:04d}: {str(e)}")
                response = ""
            
            # Write results
            pt.write({str(step).zfill(4): response})
            qt.write({str(step).zfill(4): input_text})
            
            # Handle target text (can be string or list)
            if isinstance(target_text, list):
                gt.write({str(step).zfill(4): " / ".join(target_text)})
            else:
                gt.write({str(step).zfill(4): target_text})

    logging.info(f"<========Inference completed! Results saved to {output_dir}========>")


if __name__ == "__main__":
    main()