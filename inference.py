import os
import torch
import numpy as np
import soundfile as sf
import random
import re
from pathlib import Path
from safetensors.torch import load_file


from src.utils import setup_logger, trim_silence_with_vad
from src.config import TrainConfig
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3


logger = setup_logger("Chatterbox-Inference")


cfg = TrainConfig()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_DIR = cfg.model_dir
OUTPUT_DIR = cfg.output_dir


FINETUNED_WEIGHTS = os.path.join(OUTPUT_DIR, "t3_finetuned.safetensors")
PARAMS = {
    "temperature": 0.8,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "repetition_penalty": 1.2,
    "use_phoneme": cfg.use_phoneme,
    "use_g2p": cfg.use_g2p,
    "normalize_numbers": cfg.normalize_numbers,
    "normalize_abbrev": cfg.normalize_abbrev,
}


def resolve_weights_path() -> str:
    """Resolve inference weights with fallback order: env -> final -> resume ckpt -> latest ckpt."""
    env_path = os.getenv("INFER_WEIGHTS")
    if env_path and os.path.exists(env_path):
        logger.info(f"Using weights from INFER_WEIGHTS: {env_path}")
        return env_path

    if os.path.exists(FINETUNED_WEIGHTS):
        return FINETUNED_WEIGHTS

    # Fallback to configured training resume checkpoint.
    resume_ckpt = getattr(cfg, "resume_from_checkpoint", None)
    if resume_ckpt:
        resume_model = os.path.join(resume_ckpt, "model.safetensors")
        if os.path.exists(resume_model):
            logger.warning(
                f"Final weights not found. Falling back to resume checkpoint weights: {resume_model}"
            )
            return resume_model

    # Fallback to latest checkpoint under output dir.
    ckpts = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        latest_model = ckpts[-1] / "model.safetensors"
        if latest_model.exists():
            logger.warning(
                f"Final weights not found. Falling back to latest checkpoint weights: {latest_model}"
            )
            return str(latest_model)

    raise FileNotFoundError(
        f"No inference weights found. Checked: {FINETUNED_WEIGHTS}, resume checkpoint, and latest checkpoint."
    )


def normalize_t3_state_dict_keys(state_dict: dict) -> dict:
    """Convert wrapped checkpoint keys (e.g. t3.* / module.t3.*) to plain T3 keys."""
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    normalized = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module.t3."):
            nk = nk[len("module.t3.") :]
        elif nk.startswith("t3."):
            nk = nk[len("t3.") :]
        elif nk.startswith("module."):
            nk = nk[len("module.") :]
        normalized[nk] = v

    return normalized


TEXT_TO_SAY = "Tôi tên là chấn thành đây là giọng của tôi, tôi là mộ diễ viên, tôi đang thử clone giọng "
AUDIO_PROMPT = "speaker_reference/chanthanh.wav"
OUTPUT_FILE = "./output.wav"



def load_finetuned_engine(device):
    """
    Loads the Vietnamese-only Chatterbox engine and replaces the T3 module
    with the fine-tuned version.
    """
    
    logger.info("Loading in STANDARD mode.")
    logger.info(f"Loading base model from: {BASE_MODEL_DIR}")

    EngineClass = ChatterboxTTS

    tts_engine = EngineClass.from_local(BASE_MODEL_DIR, device="cpu")
    
    # Configure New T3 Model
    logger.info(f"Initializing new T3 with vocab size: {cfg.new_vocab_size}")
    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = cfg.new_vocab_size
    t3_config.start_text_token = cfg.start_text_token
    t3_config.stop_text_token = cfg.stop_text_token
  
    new_t3 = T3(hp=t3_config)

    weights_path = resolve_weights_path()
    logger.info(f"Loading fine-tuned weights: {weights_path}")
    state_dict = load_file(weights_path, device="cpu")
    state_dict = normalize_t3_state_dict_keys(state_dict)
    new_t3.load_state_dict(state_dict, strict=True)
    logger.info("Fine-tuned weights loaded successfully.")

    tts_engine.t3 = new_t3
    
    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()

    tts_engine.device = device
    
    return tts_engine


def generate_sentence_audio(engine, text, prompt_path, **kwargs):
    """Generates audio for a single sentence and trims silence."""
    try:
        wav_tensor = engine.generate(text=text, audio_prompt_path=prompt_path, **kwargs)
        wav_np = wav_tensor.squeeze().cpu().numpy()
        trimmed_wav = trim_silence_with_vad(wav_np, engine.sr)
        return engine.sr, trimmed_wav
    except Exception as e:
        logger.error(f"Error generating sentence '{text[:30]}...': {e}")
        return 24000, np.zeros(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference running on: {device}")
    
    engine = load_finetuned_engine(device)
    
    sentences = re.split(r'(?<=[.?!])\s+', TEXT_TO_SAY.strip())
    sentences = [s for s in sentences if s.strip()]
    
    logger.info(f"Found {len(sentences)} sentences to synthesize.")
    
    all_chunks = []
    sample_rate = 24000
    
    set_seed(42)
    
    for i, sent in enumerate(sentences):
        logger.info(f"Synthesizing ({i+1}/{len(sentences)}): {sent}")
        sr, audio_chunk = generate_sentence_audio(engine, sent, AUDIO_PROMPT, **PARAMS)
        
        if len(audio_chunk) > 0:
            all_chunks.append(audio_chunk)
            sample_rate = sr
            pause_samples = int(sr * 0.2)
            all_chunks.append(np.zeros(pause_samples, dtype=np.float32))

    if all_chunks:
        final_audio = np.concatenate(all_chunks)
        sf.write(OUTPUT_FILE, final_audio, sample_rate)
        logger.info(f"Result saved to: {OUTPUT_FILE}")
    else:
        logger.error("No audio was generated.")


if __name__ == "__main__":
    main()
