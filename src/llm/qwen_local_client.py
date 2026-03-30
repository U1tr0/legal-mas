from __future__ import annotations

from typing import Any, Dict, Optional

from src.llm.base import BaseLLMClient


class QwenLocalClient(BaseLLMClient):
    def __init__(
        self,
        model_path: str,
        model_name: str = "Qwen2.5-7B-Instruct-GGUF",
        n_ctx: int = 8192,
        n_gpu_layers: int = 35,
        n_threads: int = 8,
        verbose: bool = False,
    ):
        super().__init__(model_name=model_name)
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama_cpp is not installed. Install it in the active environment to use QwenLocalClient."
            ) from e

        self.n_ctx = n_ctx
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        # JSON лучше удерживать промптом, потому что llama.cpp не всегда
        # надежно соблюдает формат структурированного ответа.
        prompt = (
            f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8")))
        available_completion_tokens = max(64, self.n_ctx - prompt_tokens - 32)
        effective_max_tokens = min(max_tokens, available_completion_tokens)

        out = self.llm(
            prompt,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.05,
            stop=["<|im_end|>"],
        )
        return out["choices"][0]["text"].strip()
