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

# Global model instance and conversation context
vocalnet_model = None
conversation_context = {}

def load_sdm():
    """
    Load VocalNet SDM (Spoken Dialogue Model) for multi-round conversation
    """
    global vocalnet_model
    
    # Set model paths
    VOCALNET_MODEL_PATH = os.environ.get('VOCALNET_MODEL', '')
    COSYVOICE_MODEL_PATH = os.environ.get('COSYVOICE_MODEL', '')
    
    if not VOCALNET_MODEL_PATH:
        raise ValueError("Please set VOCALNET_MODEL environment variable")
    if not COSYVOICE_MODEL_PATH:
        raise ValueError("Please set COSYVOICE_MODEL environment variable")
    
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


def respond_with_context(input_audio, output_path, conversation_id, round_num, conversation_history):
    """
    Generate response for input audio with conversation context
    
    Args:
        input_audio: Path to input audio file
        output_path: Path where output audio should be saved
        conversation_id: ID of the conversation
        round_num: Current round number
        conversation_history: List of previous turns in the conversation
        
    Returns:
        str: Text response from the model
    """
    global vocalnet_model
    
    if vocalnet_model is None:
        raise RuntimeError("Model not loaded. Please call load_sdm() first.")
    
    # Set output directory for audio
    output_dir = os.path.dirname(output_path)
    vocalnet_model.set_audio_dir(output_dir)
    
    # Method 1: Text-based context injection
    # Build context prompt from previous conversation history
    context_prompt = build_context_prompt(conversation_history, round_num)
    
    # Prepare input message with context
    messages = [{
        'role': 'user', 
        'content': f'<speech>{context_prompt}', 
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
            if generated_audio_path != output_path:
                shutil.move(generated_audio_path, output_path)
        else:
            logging.warning(f"No audio generated for {input_audio}, creating silent audio")
            silent_audio = torch.zeros(1, 16000)
            torchaudio.save(output_path, silent_audio, 16000)
        
        return text_response
        
    except Exception as e:
        logging.error(f"Error processing {input_audio}: {str(e)}")
        silent_audio = torch.zeros(1, 16000)
        torchaudio.save(output_path, silent_audio, 16000)
        return ""


def build_context_prompt(conversation_history, current_round):
    """
    Build context prompt from conversation history
    
    Args:
        conversation_history: List of previous conversation turns
        current_round: Current round number
        
    Returns:
        str: Context prompt to inject into the model input
    """
    if not conversation_history or current_round <= 1:
        return ""
    
    # Method 1: Simple text concatenation
    context_parts = []
    
    # Add recent conversation history (last 3 turns to avoid context overflow)
    recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
    
    for i, turn in enumerate(recent_history):
        turn_text = turn.get('source_text', '').strip()
        response_text = turn.get('output_text', '').strip()
        
        if turn_text:
            context_parts.append(f"Previous User: {turn_text}")
        if response_text:
            context_parts.append(f"Previous Assistant: {response_text}")
    
    if context_parts:
        context_prompt = "\nConversation context:\n" + "\n".join(context_parts) + "\nCurrent User: "
        return context_prompt
    
    return ""


def respond_with_memory(input_audio, output_path, conversation_id, round_num):
    """
    Alternative approach: Use global conversation memory
    
    Args:
        input_audio: Path to input audio file
        output_path: Path where output audio should be saved
        conversation_id: ID of the conversation
        round_num: Current round number
        
    Returns:
        str: Text response from the model
    """
    global vocalnet_model, conversation_context
    
    if vocalnet_model is None:
        raise RuntimeError("Model not loaded. Please call load_sdm() first.")
    
    # Initialize conversation context if not exists
    if conversation_id not in conversation_context:
        conversation_context[conversation_id] = {
            'history': [],
            'current_round': 0
        }
    
    context = conversation_context[conversation_id]
    
    # Set output directory for audio
    output_dir = os.path.dirname(output_path)
    vocalnet_model.set_audio_dir(output_dir)
    
    # Build context from memory
    context_text = ""
    if context['history']:
        # Use last 2-3 exchanges to build context
        recent_exchanges = context['history'][-6:]  # Last 3 exchanges (user + assistant)
        context_parts = []
        
        for item in recent_exchanges:
            if item['type'] == 'user':
                context_parts.append(f"User said: {item['text']}")
            elif item['type'] == 'assistant':
                context_parts.append(f"Assistant replied: {item['text']}")
        
        if context_parts:
            context_text = f"\nPrevious conversation:\n{' | '.join(context_parts)}\nNow the user says: "
    
    # Prepare input message
    messages = [{
        'role': 'user', 
        'content': f'<speech>{context_text}', 
        'path': input_audio
    }]
    
    try:
        # Call VocalNet model
        result = vocalnet_model(messages)
        text_response = result.get('text', '').strip()
        
        # Update conversation memory (you would need to get the actual user input text)
        # For now, we'll use a placeholder - in real implementation, you'd need ASR
        user_text = f"[Audio input from round {round_num}]"  # Placeholder
        
        context['history'].append({'type': 'user', 'text': user_text, 'round': round_num})
        context['history'].append({'type': 'assistant', 'text': text_response, 'round': round_num})
        context['current_round'] = round_num
        
        # Handle audio output
        if 'audio' in result and result['audio']:
            generated_audio_path = result['audio']
            if generated_audio_path != output_path:
                shutil.move(generated_audio_path, output_path)
        else:
            logging.warning(f"No audio generated for {input_audio}, creating silent audio")
            silent_audio = torch.zeros(1, 16000)
            torchaudio.save(output_path, silent_audio, 16000)
        
        return text_response
        
    except Exception as e:
        logging.error(f"Error processing {input_audio}: {str(e)}")
        silent_audio = torch.zeros(1, 16000)
        torchaudio.save(output_path, silent_audio, 16000)
        return ""


def main():
    parser = argparse.ArgumentParser(description="VocalNet multi-round inference with context for URO-Bench evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset jsonl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--context_method", type=str, default="history", choices=["history", "memory"], 
                       help="Method for handling conversation context")
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

    logging.info(f"<========Starting multi-round inference with {args.context_method} context========>")
    
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
                    # Choose context method
                    if args.context_method == "history":
                        response = respond_with_context(
                            input_path, output_path, 
                            data["id"], turn["round"], 
                            conversation_history
                        )
                    else:  # memory method
                        response = respond_with_memory(
                            input_path, output_path,
                            data["id"], turn["round"]
                        )
                    
                    logging.info(f"Conversation {data['id']}, Round {turn['round']}")
                    logging.info(f"  Input: {turn['source_text'][:100]}...")
                    logging.info(f"  Output: {response[:100]}...")
                    logging.info(f"  Audio saved to: {output_path}")
                    
                except Exception as e:
                    logging.error(f"Error processing conversation {data['id']}, round {turn['round']}: {str(e)}")
                    response = ""
                
                # Store current turn in conversation history
                turn_record = {
                    "round": turn["round"],
                    "source_wav": turn["source_wav"],
                    "source_text": turn["source_text"],
                    "target_text": turn["target_text"],
                    "output_text": response.strip(),
                }
                
                conversation.append(turn_record)
                conversation_history.append(turn_record)
            
            # Save complete conversation
            output_data = {
                "id": data["id"],
                "num_round": len(dialogue),
                "dialogue": conversation,
            }
            
            ot.write(output_data)
            logging.info(f"Completed conversation {data['id']}")

    logging.info(f"<========Multi-round inference with context completed! Results saved to {output_dir}========>")


if __name__ == "__main__":
    main()


