from typing import Union

import numpy as np
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import huggingface_config
from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy

torch.set_grad_enabled(False)

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_THRESHOLD = 0.99963529763794

BINOCULARS_FORCE_TO_CPU = os.getenv("BINOCULARS_FORCE_TO_CPU", "False").lower() in ("true", "1", "yes")

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

if BINOCULARS_FORCE_TO_CPU:
    DEVICE_1 = "cpu"
    DEVICE_2 = "cpu"

class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "HuggingFaceTB/SmolLM2-135M",
                 performer_name_or_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.threshold = BINOCULARS_THRESHOLD

        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"]
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map={"": DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    token=huggingface_config["TOKEN"]
                                                                    )

        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_token_observed = max_token_observed

    def set_threshold(self, threshold: float = None):
        if threshold is None:
            self.threshold = BINOCULARS_THRESHOLD
            return
        
        assert isinstance(threshold, float), "The threshold is not a float"
        self.threshold = threshold
    
    def get_threshold(self):
        return self.threshold
    
    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores
    
    def score_to_label(self, binoculars_scores, threshold: float = None):
        threshold = self.threshold if threshold is None else threshold
        return np.where(binoculars_scores < threshold,
                "Most likely AI-generated",
                "Most likely human-generated"
            ).tolist()

    def score_to_class(self, binoculars_scores, threshold: float = None):
        '''
        class 0 : Ai generated
        class 1 : Human generated
        '''

        threshold = self.threshold if threshold is None else threshold
        return np.where(binoculars_scores < threshold,
                0,
                1
            ).tolist()

    def predict(self, input_text: Union[list[str], str], return_fields: list[str] = None, threshold: float = None):
        '''
        Predict result AI/Human input text generated
        Return additional field in the order of the field name in `return_fields`
        '''

        # Add dynamic threshold to allow concurrent request to use the same binoculars instance
        # without changing it for all the instances
        threshold = self.threshold if threshold is None else threshold

        if return_fields is None:
            return_fields = ["label"]
        
        output = []
        binoculars_scores = np.array(self.compute_score(input_text))
        
        for field in return_fields:
            if field == "label":
                item = self.score_to_label(binoculars_scores, threshold=threshold)
            elif field == "score":
                item = binoculars_scores
            elif field == "class":
                item = self.score_to_class(binoculars_scores, threshold=threshold)
            elif field == "threshold":
                item = threshold
            else:
                raise Exception(f"Invalid field type '{field}'")

            output.append(item)
        
        if len(output) == 1:
            return output[0]
        
        return output
