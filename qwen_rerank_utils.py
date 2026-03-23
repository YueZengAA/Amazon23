# coding=utf-8
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DEFAULT_SYSTEM_TEMPLATE = (
    "Given user history in chronological order, recommend an item from the candidate pool "
    "with its index letter."
)
DEFAULT_INPUT_TEMPLATE = "User history: {}; \n Candidate pool: {}"
PROMPT_INPUT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
LABEL_SYMBOLS = [chr(code) for code in range(ord("A"), ord("Z") + 1)] + [
    chr(code) for code in range(ord("a"), ord("z") + 1)
]


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(text.split()).strip()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def read_interactions(inter_path: str) -> List[dict]:
    rows = []
    with open(inter_path, "r", encoding="utf-8") as fin:
        header = fin.readline()
        if not header:
            return rows
        for line in fin:
            user_token, history_text, pos_item_token = line.rstrip("\n").split("\t")
            history_tokens = [token for token in history_text.split(" ") if token]
            rows.append(
                {
                    "user_token": user_token,
                    "history_tokens": history_tokens,
                    "pos_item_token": pos_item_token,
                }
            )
    return rows


def load_item_titles(meta_jsonl: str, need_items: Optional[set] = None) -> Dict[str, str]:
    item2title = {}
    with open(meta_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_token = record.get("parent_asin") or record.get("asin")
            if not item_token:
                continue
            if need_items is not None and item_token not in need_items:
                continue
            if item_token in item2title:
                continue
            title = clean_text(record.get("title"))
            item2title[item_token] = title if title else item_token
    return item2title


def gather_candidate_items(candidate_paths: Sequence[str]) -> set:
    need_items = set()
    for path in candidate_paths:
        for row in iter_jsonl(path):
            need_items.update(row["history_tokens"])
            need_items.update(row["candidate_tokens"])
            need_items.add(row["pos_item_token"])
    return need_items


def truncate_title(tokenizer, title: str, max_title_len: int) -> str:
    title_tokens = tokenizer.tokenize(title)[:max_title_len]
    return tokenizer.convert_tokens_to_string(title_tokens)


def build_prompt(
    history_tokens: Sequence[str],
    candidate_tokens: Sequence[str],
    title_map: Dict[str, str],
    tokenizer,
    max_title_len: int,
    system_template: str = DEFAULT_SYSTEM_TEMPLATE,
    input_template: str = DEFAULT_INPUT_TEMPLATE,
    label: Optional[str] = None,
) -> str:
    seq_text = " \n ".join(
        f"({idx + 1}) {truncate_title(tokenizer, title_map.get(item, item), max_title_len)}"
        for idx, item in enumerate(history_tokens)
    )
    cand_text = " \n ".join(
        f"({LABEL_SYMBOLS[idx]}) {truncate_title(tokenizer, title_map.get(item, item), max_title_len)}"
        for idx, item in enumerate(candidate_tokens)
    )
    user_input = input_template.format(seq_text, cand_text)
    prompt = PROMPT_INPUT_TEMPLATE.format(instruction=system_template, input=user_input)
    if label is not None:
        prompt += label
    return prompt


def build_label_token_ids(tokenizer, num_candidates: int) -> List[int]:
    if num_candidates > len(LABEL_SYMBOLS):
        raise ValueError(f"num_candidates={num_candidates} exceeds supported symbols={len(LABEL_SYMBOLS)}")
    token_ids = []
    for symbol in LABEL_SYMBOLS[:num_candidates]:
        ids = tokenizer.encode(symbol, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Label symbol {symbol!r} is tokenized into {ids}; expected exactly one token for verbalization."
            )
        token_ids.append(ids[0])
    return token_ids


def tokenize_train_example(
    tokenizer,
    prompt_without_label: str,
    answer_symbol: str,
    max_text_len: int,
    train_on_inputs: bool,
) -> dict:
    full_prompt = prompt_without_label + answer_symbol
    tokenized_prompt = tokenizer(
        full_prompt,
        truncation=True,
        max_length=max_text_len,
        padding=False,
        return_tensors=None,
    )
    if tokenizer.eos_token_id is not None and tokenized_prompt["input_ids"][-1] != tokenizer.eos_token_id:
        tokenized_prompt["input_ids"].append(tokenizer.eos_token_id)
        tokenized_prompt["attention_mask"].append(1)

    tokenized_prompt["labels"] = tokenized_prompt["input_ids"].copy()
    if not train_on_inputs:
        prefix_ids = tokenizer(
            prompt_without_label,
            truncation=True,
            max_length=max_text_len,
            padding=False,
            return_tensors=None,
        )["input_ids"]
        prompt_len = min(len(prefix_ids), len(tokenized_prompt["labels"]))
        tokenized_prompt["labels"][:prompt_len] = [-100] * prompt_len
    return tokenized_prompt


def tokenize_eval_example(tokenizer, prompt_without_label: str, max_text_len: int, label_idx: int) -> dict:
    tokenized_prompt = tokenizer(
        prompt_without_label,
        truncation=True,
        max_length=max_text_len,
        padding=False,
        return_tensors=None,
    )
    tokenized_prompt["labels"] = label_idx
    return tokenized_prompt


def left_pad_sequences(sequences: Sequence[Sequence[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    rows = []
    for seq in sequences:
        rows.append([pad_value] * (max_len - len(seq)) + list(seq))
    return torch.tensor(rows, dtype=torch.long)


def train_collate_fn(batch: List[dict], pad_token_id: int) -> dict:
    return {
        "input_ids": left_pad_sequences([row["input_ids"] for row in batch], pad_value=pad_token_id),
        "attention_mask": left_pad_sequences([row["attention_mask"] for row in batch], pad_value=0),
        "labels": left_pad_sequences([row["labels"] for row in batch], pad_value=-100),
    }


def eval_collate_fn(batch: List[dict], pad_token_id: int, include_metadata: bool = False) -> dict:
    payload = {
        "input_ids": left_pad_sequences([row["input_ids"] for row in batch], pad_value=pad_token_id),
        "attention_mask": left_pad_sequences([row["attention_mask"] for row in batch], pad_value=0),
        "labels": torch.tensor([row["labels"] for row in batch], dtype=torch.long),
    }
    if include_metadata:
        payload.update(
            {
                "candidate_tokens": [row["candidate_tokens"] for row in batch],
                "pos_item_token": [row["pos_item_token"] for row in batch],
                "user_token": [row["user_token"] for row in batch],
                "history_tokens": [row["history_tokens"] for row in batch],
            }
        )
    return payload


def compute_recall_ndcg(scores: torch.Tensor, labels: torch.Tensor, k: int) -> Tuple[float, float]:
    topk = scores.topk(k=min(k, scores.size(1)), dim=1).indices
    hit = (topk == labels.unsqueeze(1)).any(dim=1).float()
    recall = float(hit.mean().item())
    eq = topk == labels.unsqueeze(1)
    ranks = torch.argmax(eq.int(), dim=1) + 1
    ndcg = torch.where(hit > 0, 1.0 / torch.log2(ranks.float() + 1.0), torch.zeros_like(hit))
    return recall, float(ndcg.mean().item())


def rank_candidate_scores(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sorted_scores, sorted_indices = torch.sort(scores, dim=1, descending=True)
    return sorted_scores, sorted_indices


@dataclass
class CandidateExample:
    user_token: str
    history_tokens: List[str]
    pos_item_token: str
    candidate_tokens: List[str]

    @property
    def label_idx(self) -> int:
        return self.candidate_tokens.index(self.pos_item_token)


class JsonlCandidateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        title_map: Dict[str, str],
        max_title_len: int,
        max_text_len: int,
        num_candidates: int,
        train: bool,
        train_on_inputs: bool,
        system_template: str = DEFAULT_SYSTEM_TEMPLATE,
        input_template: str = DEFAULT_INPUT_TEMPLATE,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.title_map = title_map
        self.max_title_len = max_title_len
        self.max_text_len = max_text_len
        self.num_candidates = num_candidates
        self.train = train
        self.train_on_inputs = train_on_inputs
        self.system_template = system_template
        self.input_template = input_template
        self.offsets = self._build_offsets(path)

    @staticmethod
    def _build_offsets(path: str) -> List[int]:
        offsets = []
        cursor = 0
        with open(path, "rb") as fin:
            for line in fin:
                if line.strip():
                    offsets.append(cursor)
                cursor += len(line)
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_row(self, index: int) -> dict:
        with open(self.path, "rb") as fin:
            fin.seek(self.offsets[index])
            return json.loads(fin.readline().decode("utf-8"))

    def __getitem__(self, index: int) -> dict:
        row = self._read_row(index)
        example = CandidateExample(
            user_token=row["user_token"],
            history_tokens=row["history_tokens"],
            pos_item_token=row["pos_item_token"],
            candidate_tokens=row["candidate_tokens"][: self.num_candidates],
        )
        if example.pos_item_token not in example.candidate_tokens:
            raise ValueError(f"Positive item missing from candidates at index={index} path={self.path}")
        prompt_without_label = build_prompt(
            history_tokens=example.history_tokens,
            candidate_tokens=example.candidate_tokens,
            title_map=self.title_map,
            tokenizer=self.tokenizer,
            max_title_len=self.max_title_len,
            system_template=self.system_template,
            input_template=self.input_template,
            label=None,
        )
        label_idx = example.label_idx
        if self.train:
            tokenized = tokenize_train_example(
                tokenizer=self.tokenizer,
                prompt_without_label=prompt_without_label,
                answer_symbol=LABEL_SYMBOLS[label_idx],
                max_text_len=self.max_text_len,
                train_on_inputs=self.train_on_inputs,
            )
        else:
            tokenized = tokenize_eval_example(
                tokenizer=self.tokenizer,
                prompt_without_label=prompt_without_label,
                max_text_len=self.max_text_len,
                label_idx=label_idx,
            )
            tokenized["candidate_tokens"] = example.candidate_tokens
            tokenized["pos_item_token"] = example.pos_item_token
            tokenized["user_token"] = example.user_token
            tokenized["history_tokens"] = example.history_tokens
        return tokenized


def summarize_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    recall10, ndcg10 = compute_recall_ndcg(scores, labels, k=10)
    return {
        "recall_at_10": recall10,
        "ndcg_at_10": ndcg10,
        "recall@10": recall10,
        "ndcg@10": ndcg10,
    }
