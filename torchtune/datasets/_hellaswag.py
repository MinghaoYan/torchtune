# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Mapping, Optional

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import Tokenizer

from torchtune.data import (
    InstructTemplate,
)
from functools import partial
from torchtune.config._utils import _get_component_from_path

class HellaswagDataset(Dataset):
    """
    Freeform dataset for any unstructured text corpus. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an ``encode`` and ``decode`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        column (str): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data. For local datasets with a single column, use the default "text",
            which is what is assigned by Hugging Face datasets when loaded into memory. Default is "text".
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        column_map: Optional[Dict[str, str]] = None,
        max_seq_len: Optional[int] = None,
        template: InstructTemplate = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.max_seq_len = max_seq_len
        self._column_map = column_map
        self.template = template

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = self.template.format(sample, self._column_map)
        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=True)

        # Truncate if needed, but don't coerce EOS id
        if self.max_seq_len is not None:
            tokens = truncate(tokens, self.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def hellaswag_dataset(
    tokenizer: Tokenizer,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    template: str = "torchtune.data.HellaswagTemplate",
    **load_dataset_kwargs: Dict[str, Any],
) -> HellaswagDataset:
    """
    """

    split = load_dataset_kwargs.pop('split', "train")

    ds = HellaswagDataset(
        tokenizer=tokenizer,
        source=source,
        column_map=column_map,
        max_seq_len=max_seq_len,
        template=_get_component_from_path(template),
        split=split,
    )
    return PackedDataset(ds, max_seq_len=max_seq_len) if packed else ds


hellaswag_dataset = partial(hellaswag_dataset, source="Rowan/hellaswag")
