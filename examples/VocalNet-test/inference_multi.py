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
    Load VocalNet SDM (Spoken Dialogue Model) for multi-round conversation
    """
    global vocalnet_model
    
    # Set model paths - you need to modify these paths according to your setup
    VOCALNET_MODEL_PATH = os.environ.get('VOCALNET_MODEL', '')
    COSYVOICE_MODEL_PATH = os.environ.get('COSYVOICE_MODEL', '')
    
    if not VOCALNET_MODEL_PATH:
        raise ValueError("Please set VOCALNET_MODEL_PATH environment variable")
    if not COSYVOICE_MODEL_PATH:
        raise ValueError("Please set COSYVOICE_MODEL_PATH environment variable")
    
    logging.info("Initializing VocalNet model for multi-round conversation...")
    
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


def build_history_pairs(conversation_history, max_turns=5):
    """
    Convert conversation history into a list of {from, value} pairs expected by the model preprocessor.

    Example:
    [
        {"from": "human", "value": "asr results"},
        {"from": "gpt",   "value": "assistant text"},
        ...
    ]
    """
    if not conversation_history:
        return []

    recent_history = (
        conversation_history[-max_turns:]
        if len(conversation_history) > max_turns
        else conversation_history
    )

    pairs = []
    for turn in recent_history:
        user_asr_text = (turn.get("asr_text", "") or turn.get("source_text", "")).strip()
        assistant_text = (turn.get("output_text", "")).strip()

        if user_asr_text:
            pairs.append({"from": "human", "value": user_asr_text})
        if assistant_text:
            pairs.append({"from": "gpt", "value": assistant_text})

    return pairs


def get_asr_transcription(audio_path):
    """
    Get ASR transcription of audio file using Whisper
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        str: Transcribed text
    """
    try:
        import whisper
        
        # Load Whisper model (cache it for efficiency)
        if not hasattr(get_asr_transcription, 'model'):
            whisper_model_path = os.environ.get('WHISPER_MODEL_PATH', '/home/tuwenming/Models/openai/whisper-large-v3')
            logging.info(f"Loading Whisper model from {whisper_model_path}...")
            
            if os.path.exists(whisper_model_path):
                # Load local model
                get_asr_transcription.model = whisper.load_model(f"{whisper_model_path}/large_v3.pt", device="cuda")
            else:
                # Load from online
                get_asr_transcription.model = whisper.load_model("large-v3")
        
        # Transcribe audio
        result = get_asr_transcription.model.transcribe(audio_path)
        return result["text"].strip()
        
    except Exception as e:
        logging.warning(f"ASR transcription failed for {audio_path}: {str(e)}")
        return ""


def respond(input_audio, output_path, conversation_history=None):
    """
    Generate response for input audio with conversation context
    
    Args:
        input_audio: Path to input audio file
        output_path: Path where output audio should be saved
        conversation_history: List of previous conversation turns with ASR text and assistant responses
        
    Returns:
        str: Text response from the model
    """
    global vocalnet_model
    
    if vocalnet_model is None:
        raise RuntimeError("Model not loaded. Please call load_sdm() first.")
    
    # Set output directory for audio
    output_dir = os.path.dirname(output_path)
    vocalnet_model.set_audio_dir(output_dir)
    
    # Build structured history pairs for multi-round conversation
    history_pairs = build_history_pairs(conversation_history)

    # Prepare input message with structured history for the model to consume
    base_message = {
        'role': 'user',
        'content': '<speech>',
        'path': input_audio
    }
    if history_pairs:
        base_message['history'] = history_pairs
    messages = [base_message]
    
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
    parser = argparse.ArgumentParser(description="VocalNet multi-round inference for URO-Bench evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset jsonl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Define output file
    output_text = os.path.join(output_dir, "output_with_text.jsonl")

    logging.info("<========Loading VocalNet model========>")
    load_sdm()

    logging.info("<========Starting multi-round inference========>")
    
    # Count total samples for progress bar
    with open(args.dataset, "r") as f:
        total_samples = sum(1 for _ in jsonlines.Reader(f))
    
    with open(args.dataset, "r") as f, \
         jsonlines.open(output_text, mode="w") as ot:
        
        for data in tqdm(jsonlines.Reader(f), total=total_samples, desc="Processing conversations"):
            dialogue = data["dialogue"]
            conversation = []
            
            # Create directory for this conversation
            id_dir = os.path.join(output_dir, str(data["id"]))
            os.makedirs(id_dir, exist_ok=True)
            
            logging.info(f"Processing conversation {data['id']} with {len(dialogue)} rounds")
            
            # Track conversation history for context
            conversation_history = []
            
            for turn in dialogue:
                input_path = os.path.join(
                    os.path.dirname(args.dataset), turn["source_wav"]
                )
                output_path = os.path.join(id_dir, f"chat_{turn['round']}.wav")
                
                try:
                    # Get ASR transcription of current user input
                    logging.info(f"Getting ASR transcription for round {turn['round']}...")
                    current_asr_text = get_asr_transcription(input_path)
                    
                    # Generate response with conversation history
                    logging.info(f"Generating response with context (history length: {len(conversation_history)})...")
                    response = respond(input_path, output_path, conversation_history)
                    
                    logging.info(f"Conversation {data['id']}, Round {turn['round']}")
                    logging.info(f"  Input (Original): {turn['source_text'][:100]}...")
                    logging.info(f"  Input (ASR): {current_asr_text[:100]}...")
                    logging.info(f"  Output: {response[:100]}...")
                    logging.info(f"  Audio saved to: {output_path}")
                    
                except Exception as e:
                    logging.error(f"Error processing conversation {data['id']}, round {turn['round']}: {str(e)}")
                    response = ""
                    current_asr_text = turn.get("source_text", "")
                
                # Store current conversation turn
                current_turn = {
                    "round": turn["round"],
                    "source_wav": turn["source_wav"],
                    "source_text": turn["source_text"],
                    "target_text": turn["target_text"],
                    "asr_text": current_asr_text,
                    "output_text": response.strip(),
                }
                
                conversation.append(current_turn)
                
                # Add to conversation history for next turn's context
                conversation_history.append(current_turn)
            
            # Save complete conversation
            output_data = {
                "id": data["id"],
                "num_round": len(dialogue),
                "dialogue": conversation,
            }
            
            ot.write(output_data)
            logging.info(f"Completed conversation {data['id']}")

    logging.info(f"<========Multi-round inference completed! Results saved to {output_dir}========>")


if __name__ == "__main__":
    main()