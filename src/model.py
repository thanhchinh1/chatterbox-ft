import torch
import torch.nn.functional as F
from src.chatterbox_.models.t3.modules.cond_enc import T3Cond
from src.config import TrainConfig
from src.utils import setup_logger


logger = setup_logger(__name__)


def resize_and_load_t3_weights(new_model: torch.nn.Module, pretrained_state_dict: dict):
    """
    Loads pretrained weights into a new T3 model with a different vocabulary size.
    Features: Initialize new tokens with the AVERAGE of existing tokens.
    """
    new_model_state_dict = new_model.state_dict()

    embedding_layer_name = "text_emb.weight"
    output_head_name = "text_head.weight"

    # Step 1: Copy weights for ALL matching layers
    for name, param in pretrained_state_dict.items():
        
        if name not in [embedding_layer_name, output_head_name]:
            
            if name in new_model_state_dict and new_model_state_dict[name].shape == param.shape:
                new_model_state_dict[name].copy_(param)
                
            else:
                logger.warning(f"Layer skipped (mismatch): {name}")


    # Step 2: Smart copy for Embedding Layer (Average Init)
    if embedding_layer_name in pretrained_state_dict:
        
        old_emb_weights = pretrained_state_dict[embedding_layer_name]
        old_vocab_size, _ = old_emb_weights.shape
        new_vocab_size = new_model_state_dict[embedding_layer_name].shape[0]

        # A) Copy old weights
        new_model_state_dict[embedding_layer_name][:old_vocab_size, :].copy_(old_emb_weights)
        logger.info(f"Embedding layer: {old_vocab_size} tokens preserved.")

        # B) Initialize new tokens with average
        if new_vocab_size > old_vocab_size:
            
            mean_emb = old_emb_weights.mean(dim=0)
            num_new_tokens = new_vocab_size - old_vocab_size
            
            new_model_state_dict[embedding_layer_name][old_vocab_size:, :].copy_(mean_emb.unsqueeze(0).expand(num_new_tokens, -1))
            
            logger.info(f"Embedding layer: {num_new_tokens} new tokens initialized with mean.")


    # Step 3: Smart copy for Output Head (Average Init)
    if output_head_name in pretrained_state_dict:
        
        old_head_weights = pretrained_state_dict[output_head_name]
        old_vocab_size, _ = old_head_weights.shape
        new_vocab_size = new_model_state_dict[output_head_name].shape[0]

        # A) Copy old weights
        new_model_state_dict[output_head_name][:old_vocab_size, :].copy_(old_head_weights)
        logger.info(f"Output head: {old_vocab_size} tokens preserved.")

        # B) Initialize new neurons with average
        if new_vocab_size > old_vocab_size:
            
            mean_head = old_head_weights.mean(dim=0)
            num_new_tokens = new_vocab_size - old_vocab_size
            new_model_state_dict[output_head_name][old_vocab_size:, :].copy_(mean_head.unsqueeze(0).expand(num_new_tokens, -1))
            
            logger.info(f"Output head: {num_new_tokens} new neurons initialized with mean.")

    # Step 4: Load the updated state dict into the new model
    new_model.load_state_dict(new_model_state_dict)
    logger.info("All weights transferred successfully (Mean Initialization applied)!")

    return new_model


class ChatterboxTrainerWrapper(torch.nn.Module):
    """
    Wrapper class to calculate Loss inside the Forward pass for HuggingFace Trainer.
    """
    
    def __init__(self, t3_model):
        
        super().__init__()
        self.t3 = t3_model
        
        self.cfg = TrainConfig()
        
        if hasattr(t3_model.hp, 'speech_cond_prompt_len'):
            self.prompt_token_len = t3_model.hp.speech_cond_prompt_len
        else:
            self.prompt_token_len = 150 


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.t3.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.t3.get_input_embeddings()


    def _normalize_text_boundary_tokens(self, text_tokens: torch.Tensor, text_token_lens: torch.Tensor):
        """
        Data-parallel batches can occasionally carry malformed text boundaries.
        Ensure each valid sequence contains both start/stop text tokens expected by T3.
        """
        start_id = self.t3.hp.start_text_token
        stop_id = self.t3.hp.stop_text_token

        text_tokens = text_tokens.clone()
        text_token_lens = text_token_lens.to(dtype=torch.long)
        B = text_tokens.size(0)
        T = text_tokens.size(1)

        if T == 0:
            return text_tokens

        # Safety clamp against occasional corrupted lens values on parallel workers.
        text_token_lens = torch.clamp(text_token_lens, min=1, max=T)

        # Hard guarantee: every sequence starts with start token.
        text_tokens[:, 0] = start_id

        for i in range(B):
            seq_len = int(text_token_lens[i].item())
            seq_len = max(1, min(seq_len, T))

            seq = text_tokens[i, :seq_len]
            if not (seq == stop_id).any():
                text_tokens[i, seq_len - 1] = stop_id

        return text_tokens


    def _validate_token_ranges(self, text_tokens: torch.Tensor, speech_tokens: torch.Tensor, prompt_tokens: torch.Tensor):
        text_vocab = self.t3.text_emb.num_embeddings
        speech_vocab = self.t3.speech_emb.num_embeddings

        tmin = int(text_tokens.min().item())
        tmax = int(text_tokens.max().item())
        smin = int(speech_tokens.min().item())
        smax = int(speech_tokens.max().item())
        pmin = int(prompt_tokens.min().item())
        pmax = int(prompt_tokens.max().item())

        if tmin < 0 or tmax >= text_vocab:
            raise ValueError(
                f"text_tokens out of range: min={tmin}, max={tmax}, vocab={text_vocab}"
            )

        if smin < 0 or smax >= speech_vocab:
            raise ValueError(
                f"speech_tokens out of range: min={smin}, max={smax}, vocab={speech_vocab}"
            )

        if pmin < 0 or pmax >= speech_vocab:
            raise ValueError(
                f"prompt_tokens out of range: min={pmin}, max={pmax}, vocab={speech_vocab}"
            )


    def forward(
            self,
            text_tokens, 
            text_token_lens,
            speech_tokens, 
            speech_token_lens,
            speaker_emb, 
            prompt_tokens,
            prompt_lens=None):

        # Defensive sanitization for occasional malformed lengths in multi-worker / multi-GPU paths.
        text_token_lens = text_token_lens.to(dtype=torch.long)
        speech_token_lens = speech_token_lens.to(dtype=torch.long)
        text_token_lens = torch.clamp(text_token_lens, min=1, max=text_tokens.size(1))
        speech_token_lens = torch.clamp(speech_token_lens, min=1, max=speech_tokens.size(1))
        if prompt_lens is not None:
            prompt_lens = prompt_lens.to(dtype=torch.long)
            prompt_lens = torch.clamp(prompt_lens, min=0, max=prompt_tokens.size(1))

        text_tokens = self._normalize_text_boundary_tokens(text_tokens, text_token_lens)
        self._validate_token_ranges(text_tokens, speech_tokens, prompt_tokens)

        device = text_tokens.device
        batch_size = text_tokens.size(0)
        
        emotion_adv = 0.5 * torch.ones(batch_size, 1, 1).to(device)
        
        t3_cond = T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=prompt_tokens,
            emotion_adv=emotion_adv
        )

        # Forward Pass
        out = self.t3.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True
        )

        IGNORE_ID = -100

        speech_logits = out.speech_logits[:, :-1, :].transpose(1, 2)
        speech_labels = speech_tokens[:, 1:] 
   
        curr_speech_len = speech_labels.size(1)
        mask_speech_pad = torch.arange(curr_speech_len, device=device)[None, :] >= (speech_token_lens[:, None] - 1)


        if prompt_lens is not None:
            mask_prompt = torch.arange(curr_speech_len, device=device)[None, :] < prompt_lens[:, None]
        else:
            # Standard collator path: prompt is padded to fixed width for the whole batch.
            mask_prompt = torch.arange(curr_speech_len, device=device)[None, :] < prompt_tokens.size(1)


        speech_labels = speech_labels.masked_fill(mask_speech_pad | mask_prompt, IGNORE_ID)

        loss_speech = F.cross_entropy(speech_logits, speech_labels, ignore_index=IGNORE_ID)


        text_logits = out.text_logits[:, :-1, :].transpose(1, 2)
        text_labels = text_tokens[:, 1:]
            
        curr_text_len = text_labels.size(1)
        mask_text_pad = torch.arange(curr_text_len, device=device)[None, :] >= (text_token_lens[:, None] - 1)
        
        text_labels = text_labels.masked_fill(mask_text_pad, IGNORE_ID)
            
        loss_text = F.cross_entropy(text_logits, text_labels, ignore_index=IGNORE_ID)

        total_loss = loss_text + loss_speech


        return (total_loss, None)