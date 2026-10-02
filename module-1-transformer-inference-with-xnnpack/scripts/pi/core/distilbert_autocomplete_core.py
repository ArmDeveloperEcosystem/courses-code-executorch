# ExecuTorch fill-mask inference used by the Module 1 autocomplete demo.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


MODULE1_SCRIPTS = Path(__file__).resolve().parents[2]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from _support.common import configure_quiet_output, load_executorch_program, suppress_stderr


WORD_RE = re.compile(r"^[A-Za-z]+$")


@dataclass(frozen=True)
class FillSuggestion:
    token: str
    sentence: str
    score: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "token": self.token,
            "sentence": self.sentence,
            "score": self.score,
        }


@dataclass(frozen=True)
class FillMaskResult:
    suggestions: list[FillSuggestion]
    latency_ms: float

    def as_dict(self) -> dict[str, object]:
        return {
            "suggestions": [suggestion.as_dict() for suggestion in self.suggestions],
            "latency_ms": self.latency_ms,
        }


@dataclass
class DistilBertAutocompleteRunner:
    # Turn fill-mask vocabulary logits into readable word suggestions.
    spec: dict[str, Any]
    tokenizer: Any
    model: Any
    startup: dict[str, str | float | int]

    @property
    def mask_token(self) -> str:
        return str(self.tokenizer.mask_token)

    def suggest(self, prompt: str, *, topk: int) -> FillMaskResult:
        # Append one mask, run the .pte, and decode the highest-scoring words.
        import torch

        started = time.perf_counter()
        text = prompt.replace(self.mask_token, "").strip()
        if not text:
            return FillMaskResult(suggestions=[], latency_ms=0.0)
        masked_prompt = self.mask_at_end(text)

        encoded = self.tokenizer(
            masked_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=int(self.spec["seq_len"]),
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # ExecuTorch Python runtime: invoke the exported forward method.
        with suppress_stderr():
            logits = self.model.forward((input_ids, attention_mask))[0]

        scores = torch.softmax(logits[0], dim=-1)
        # Inspect more logits than the requested count because subword pieces
        # and special tokens are filtered out below.
        candidate_count = min(max(50, topk * 8), int(scores.numel()))
        top = torch.topk(scores, candidate_count)
        mask_idx = (input_ids[0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

        suggestions = []
        for token_id, score in zip(top.indices.tolist(), top.values.tolist()):
            if len(suggestions) >= max(1, topk):
                break
            token_piece = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            filled = input_ids.clone()
            filled[0, mask_idx] = token_id
            token = self.tokenizer.decode([token_id]).strip()
            if not is_word_suggestion(token, token_piece, self.tokenizer.all_special_tokens):
                continue
            sentence = self.tokenizer.decode(filled[0], skip_special_tokens=True)
            suggestions.append(FillSuggestion(token=token, sentence=sentence, score=float(score)))

        return FillMaskResult(
            suggestions=suggestions,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def mask_at_end(self, text: str) -> str:
        # Append the mask token when the supplied text does not end with one.
        if text.endswith(self.mask_token):
            return text
        return f"{text.rstrip()} {self.mask_token}"


def is_word_suggestion(token: str, token_piece: str, special_tokens: list[str]) -> bool:
    # Reject special tokens, continuation pieces, and non-word candidates.
    if token_piece in special_tokens or token_piece.startswith("##"):
        return False
    return bool(WORD_RE.fullmatch(token))


def load_distilbert_autocomplete_runner(spec_path: Path, pte_path: Path) -> DistilBertAutocompleteRunner:
    # Load the tokenizer, model contract, and ExecuTorch program once at startup.
    configure_quiet_output(include_transformers=True)

    from transformers import AutoTokenizer

    load_started = time.perf_counter()
    spec = json.loads(spec_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], use_fast=True)
    model = load_executorch_program(pte_path)

    startup = {
        "model": spec["model_id"],
        "backend": pte_path.stem,
        "seq_len": int(spec["seq_len"]),
        "load_ms": (time.perf_counter() - load_started) * 1000.0,
    }
    return DistilBertAutocompleteRunner(spec=spec, tokenizer=tokenizer, model=model, startup=startup)
