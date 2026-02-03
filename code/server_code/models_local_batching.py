import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

@dataclass
class SimpleResponse:
    content: str

class LocalChatModel:
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_new_tokens: int = 140,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True,
        )

        # REQUIRED FOR BATCHING: Left-padding is necessary for decoder-only models
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": "cuda:0"}, # Maps to the GPU selected in bash
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _build_prompts(self, texts: list[str]) -> list[str]:
        """Wraps a list of strings in the model's chat template."""
        formatted_prompts = []
        for text in texts:
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": text}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
            else:
                formatted_prompts.append(f"User: {text}\nAssistant:")
        return formatted_prompts

    @torch.inference_mode()
    def invoke_batch(self, user_texts: list[str]) -> list[SimpleResponse]:
        """Processes multiple prompts at once on the GPU."""
        prompts = self._build_prompts(user_texts)
        
        # Tokenize all prompts with padding
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        input_length = inputs.input_ids.shape[1]

        do_sample = self.temperature > 0
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        responses = []
        for i in range(len(prompts)):
            # Extract only the newly generated tokens for each item in the batch
            decoded = self.tokenizer.decode(out[i][input_length:], skip_special_tokens=True)
            responses.append(SimpleResponse(content=decoded.strip()))
        
        return responses

    def invoke(self, messages):
        """Maintains compatibility with your original single-call structure."""
        m0 = messages[0]
        user_text = m0.content if hasattr(m0, "content") else str(m0)
        return self.invoke_batch([user_text])[0]