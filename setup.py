import os
import requests
import sys
import json
import shutil
from tqdm import tqdm
from src.config import TrainConfig


DEST_DIR = "pretrained_models"
 
CHATTERBOX_FILES = {
    "ve.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.safetensors?download=true",
    "t3_cfg.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_cfg.safetensors?download=true",
    "s3gen.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.safetensors?download=true",
    "conds.pt": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/conds.pt?download=true",
    "tokenizer.json": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/grapheme_mtl_merged_expanded_v1.json?download=true"
}

VIETNAMESE_TOKENIZER_SOURCE = "tokenizer.json"

def download_file(url, dest_path):
    """Downloads a file from a URL to a specific destination with a progress bar."""
    
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading: {os.path.basename(dest_path)}...")
    
    try:
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            
            for data in response.iter_content(block_size):
                
                size = file.write(data)
                bar.update(size)
                
        print(f"Download complete: {dest_path}\n")
        
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)



def apply_vietnamese_only_tokenizer(dest_dir: str) -> int:
    """
    Replaces tokenizer.json under pretrained models with the repository's
    Vietnamese-only tokenizer definition and returns its vocab size.
    """
    if not os.path.exists(VIETNAMESE_TOKENIZER_SOURCE):
        print(
            f"ERROR: Vietnamese tokenizer source not found: {VIETNAMESE_TOKENIZER_SOURCE}"
        )
        sys.exit(1)

    destination = os.path.join(dest_dir, "tokenizer.json")
    shutil.copyfile(VIETNAMESE_TOKENIZER_SOURCE, destination)

    with open(destination, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    vocab_size = len(tokenizer_data.get("model", {}).get("vocab", {}))
    if vocab_size == 0:
        print("ERROR: Vietnamese tokenizer vocab is empty.")
        sys.exit(1)

    print("\n--- Vietnamese-only Tokenizer Applied ---")
    print(f"Source: {VIETNAMESE_TOKENIZER_SOURCE}")
    print(f"Destination: {destination}")
    print(f"Vocab size: {vocab_size}")
    return vocab_size




def main():
    
    print("--- Chatterbox Pretrained Model Setup ---\n")
    
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        
        print(f"Creating directory: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)
        
    else:
        print(f"Directory found: {DEST_DIR}")
        

    cfg = TrainConfig()

    print(f"Mode: CHATTERBOX-TTS (Standard only, {len(CHATTERBOX_FILES)} files)")
    FILES_TO_DOWNLOAD = CHATTERBOX_FILES

    # 2. Download files
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = os.path.join(DEST_DIR, filename)
        download_file(url, dest_path)

    vn_vocab_size = apply_vietnamese_only_tokenizer(DEST_DIR)
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE (VIETNAMESE-ONLY MODE)")
    print("All models are set up in 'pretrained_models/' folder.")
    print("Please set 'new_vocab_size' in 'src/config.py' to:")
    print(f" {vn_vocab_size}")
    print("="*60 + "\n")



if __name__ == "__main__":
    main()
