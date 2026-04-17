import os
import sys
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file

from src.config import TrainConfig
from src.dataset import ChatterboxDataset, data_collator_standart
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.preprocess_ljspeech import preprocess_dataset_ljspeech
from src.utils import setup_logger, check_pretrained_models

from src.inference_callback import InferenceCallback

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxFinetune")


def main():
    
    cfg = TrainConfig()

    if cfg.force_single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.single_gpu_id)
        logger.info(f"force_single_gpu enabled -> CUDA_VISIBLE_DEVICES={cfg.single_gpu_id}")
    
    logger.info("--- Starting Chatterbox Finetuning ---")
    logger.info("Mode: CHATTERBOX-TTS (Standard only)")

    # 0. CHECK MODEL FILES
    if not check_pretrained_models(mode="chatterbox"):
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 1. Standard-only engine class
    EngineClass = ChatterboxTTS
    
    logger.info(f"Device: {device}")
    logger.info(f"Model Directory: {cfg.model_dir}")

    # 2. LOAD ORIGINAL MODEL TEMPORARILY
    logger.info("Loading original model to extract weights...")
    # Loading on CPU first to save VRAM
    tts_engine_original = EngineClass.from_local(cfg.model_dir, device="cpu")

    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    # 3. CREATE NEW T3 MODEL WITH NEW VOCAB SIZE
    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")
    
    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size
    new_t3_config.start_text_token = cfg.start_text_token
    new_t3_config.stop_text_token = cfg.stop_text_token

    # We prevent caching during training.
    if hasattr(new_t3_config, "use_cache"):
        new_t3_config.use_cache = False
    else:
        setattr(new_t3_config, "use_cache", False)

    new_t3_model = T3(hp=new_t3_config)

    # 4. TRANSFER WEIGHTS
    logger.info("Transferring weights...")
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)


    # Clean up memory
    del tts_engine_original
    del pretrained_t3_state_dict

    # 5. PREPARE ENGINE FOR TRAINING
    # Reload engine components (VoiceEncoder, S3Gen) but inject our new T3
    tts_engine_new = EngineClass.from_local(cfg.model_dir, device="cpu")
    tts_engine_new.t3 = new_t3_model 

    # Freeze other components
    logger.info("Freezing S3Gen and VoiceEncoder...")
    for param in tts_engine_new.ve.parameters(): 
        param.requires_grad = False
        
    for param in tts_engine_new.s3gen.parameters(): 
        param.requires_grad = False

    # Enable Training for T3
    tts_engine_new.t3.train()
    for param in tts_engine_new.t3.parameters(): 
        param.requires_grad = True

    if cfg.preprocess:
        logger.info("Initializing LJSpeech preprocessing...")
        preprocess_dataset_ljspeech(cfg, tts_engine_new)
      
    else:
        logger.info("Skipping the preprocessing dataset step...")
            
        
    # 6. DATASET & WRAPPER
    logger.info("Initializing Dataset...")
    train_ds = ChatterboxDataset(cfg)
    
    
    trainer_callbacks = []
    if cfg.is_inference:
        inference_cb = InferenceCallback(cfg)
        trainer_callbacks.append(inference_cb)
    
    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)


    logger.info("Using Standard Data Collator")
    selected_collator = data_collator_standart


    # 7. TRAINING ARGUMENTS
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        warmup_ratio=cfg.warmup_ratio,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_strategy="epoch",
        remove_unused_columns=False, # Required for our custom wrapper
        dataloader_num_workers=cfg.dataloader_num_workers,    
        report_to=["tensorboard"],
        fp16=use_fp16,
        bf16=use_bf16,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=True, # This setting theoretically reduces VRAM usage by 60%.
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=selected_collator,
        callbacks=trainer_callbacks
    )

    logger.info("Starting Training Loop...")
    resume_path = cfg.resume_from_checkpoint if getattr(cfg, "resume_from_checkpoint", None) else None
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming training from checkpoint: {resume_path}")
    elif resume_path:
        logger.warning(f"resume_from_checkpoint not found: {resume_path}. Starting from scratch.")
        resume_path = None

    trainer.train(resume_from_checkpoint=resume_path)


    # 8. SAVE FINAL MODEL
    logger.info("Training complete. Saving model...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    filename = "t3_finetuned.safetensors"
    final_model_path = os.path.join(cfg.output_dir, filename)

    save_file(tts_engine_new.t3.state_dict(), final_model_path)
    logger.info(f"Model saved to: {final_model_path}")


if __name__ == "__main__": 
    main()
