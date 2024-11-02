import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def get_device():
  """Determines the available device: GPU, MPS, or CPU."""
  if torch.cuda.is_available():
    return torch.device('cuda')
  elif torch.backends.mps.is_available():
    return torch.device('mps')
  else:
    return torch.device('cpu')


class GrammarCorrector:
  def __init__(self, device=None):
    self.device = device or get_device()

    # Models
    peft_model_path = "akhmat-s/t5-large-quant-grammar-corrector"
    base_model_path = "akhmat-s/t5-base-grammar-corrector"

    # Load models and tokenizers
    self.model_large, self.tokenizer_large = self._load_peft_model(peft_model_path)
    self.model_base, self.tokenizer_base = self._load_model(base_model_path)

    # Generation configurations
    self.configurations = [
      {'temperature': 0.5, 'top_k': 20, 'top_p': 0.5, 'no_repeat_ngram_size': 2},
      {'temperature': 1.0, 'top_k': 50, 'top_p': 1.0, 'no_repeat_ngram_size': 0},
    ]

    # Prompts
    self.prompts = [
      'grammar:',
      'Correct the grammar:',
    ]

  def _load_peft_model(self, model_path):
    """Loads a quantized model and its tokenizer."""
    config = PeftConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, model_path)
    model.to(self.device)
    return model, tokenizer

  def _load_model(self, model_path):
    """Loads a standard model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(self.device)
    return model, tokenizer

  @staticmethod
  def _calculate_similarity(text_a, text_b):
    """Calculates similarity between two texts using SequenceMatcher."""
    return SequenceMatcher(None, text_a, text_b).ratio()

  def _model_internal_voting(self, model, tokenizer, input_text):
    """Generates samples and performs internal voting to select the best output."""
    samples = []
    for config in self.configurations:
      for prompt in self.prompts:
        full_input = f"{prompt} {input_text}"
        inputs = tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        # Generate output with the current configuration
        with torch.no_grad():
          outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512,
            do_sample=True,
            temperature=config.get('temperature', 1.0),
            top_k=config.get('top_k', 50),
            top_p=config.get('top_p', 1.0)
          )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        samples.append(generated_text)

    num_samples = len(samples)
    scores = [0.0] * num_samples

    # Calculate cumulative similarity between samples
    for i in range(num_samples):
      for j in range(i + 1, num_samples):
        similarity = self._calculate_similarity(samples[i], samples[j])
        scores[i] += similarity
        scores[j] += similarity

    # Select the sample with the highest cumulative similarity
    best_index = scores.index(max(scores))
    best_output = samples[best_index]

    return best_output

  def correct(self, input_text):
    """Corrects grammar and punctuation in the given text using sequential processing."""
    # Grammar correction
    intermediate_result = self._model_internal_voting(self.model_large, self.tokenizer_large, input_text)

    # Punctuation correction
    final_result = self._model_internal_voting(self.model_base, self.tokenizer_base, intermediate_result)

    return final_result
