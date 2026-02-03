import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


@dataclass
class SimpleResponse:
    content: str


class LocalChatModel:
    """
    Minimal wrapper that mimics:
      model.invoke([HumanMessage("...")]).content

    Uses HF Transformers + bitsandbytes 4-bit on cuda:0 (correct under Slurm masking).
    """

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

        import os
        if not os.path.isdir(model_path):
            raise RuntimeError(f"Model path does not exist or is not a directory: {model_path}")


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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            # This change allows it to use whatever GPU is "visible"
            device_map="auto", 
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )


        self.model.eval()

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_prompt(self, user_text: str) -> str:
        # Prefer the model's chat template (Llama/Gemma have this)
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_text}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {user_text}\nAssistant:"

    @torch.inference_mode()
    def invoke(self, messages):
        # messages is like [HumanMessage("...")]
        m0 = messages[0]
        user_text = m0.content if hasattr(m0, "content") else str(m0)

        prompt = self._build_prompt(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

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

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Conservative extraction of the assistant continuation
        content = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
        return SimpleResponse(content=content)
