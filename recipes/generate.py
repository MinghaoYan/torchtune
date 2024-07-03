# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn

from torchtune import config, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message

logger = utils.get_logger("DEBUG")

from torch.utils.data import DataLoader, DistributedSampler
from torchtune.datasets import ConcatDataset
from functools import partial

class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        combined_state_dict = {**ckpt_dict[utils.MODEL_KEY], **ckpt_dict[utils.ADAPTER_KEY]}
        # print(ckpt_dict)
        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=combined_state_dict,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)
        # print(model_state_dict)
        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(batch_size=1, dtype=self._dtype)

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Union[DictConfig, str],
        chat_format: Optional[ChatFormat],
        instruct_template: Optional[InstructTemplate],
    ) -> List[Message]:
        """
        Either:
        (1) a raw string is passed as the prompt, in which case we call tokenizer.encode directly, or
        (2) a DictConfig is passed as the prompt. In this case there are three possibilities:
            (a) an InstructTemplate is provided. Since instruct templates output a string, we will
                call tokenizer.encode on the output of the instruct template.
            (b) a ChatFormat is provided. Since chat formats output a list of messages, we will
                call tokenizer.tokenize_messages on the output of the chat format.
            (c) neither an InstructTemplate nor a ChatFormat is provided. In this case we will
                convert the DictConfig to a list of messages and call tokenizer.tokenize_messages directly.
        """

        # Should only be chat-style prompt or instruct-style prompt
        if chat_format and instruct_template:
            raise ValueError(
                "Cannot pass both chat format and instruct template for generation"
            )

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self._tokenizer.encode(prompt, add_bos=True, add_eos=False)

        # dict.items() will respect order for Python >= 3.7
        else:
            messages = [Message(role=k, content=v) for k, v in prompt.items()]
            messages += [Message(role="assistant", content="")]
            if chat_format:
                chat_format = _get_component_from_path(chat_format)
                messages = chat_format.format(messages)
            return self._tokenizer.tokenize_messages(messages)[0]
        
    
    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
            )
            if not packed
            else None,
        )

        logger.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        # tokens = self.convert_prompt_to_tokens(
        #     cfg.prompt, cfg.get("chat_format", None), cfg.get("instruct_template", None)
        # )
        # prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        for _, batch in enumerate(self._dataloader):

            # Both are shape [b, s]
            tokens = batch["tokens"].to(self._device)

            custom_generate_next_token = None

            t0 = time.perf_counter()
            generated_tokens = utils.generate(
                model=self._model,
                prompt=tokens,
                max_generated_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                pad_id=self._tokenizer.pad_id,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0

            logger.info(self._tokenizer.decode(generated_tokens[0]))

            model_size = sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(
                        self._model.parameters(), self._model.buffers()
                    )
                ]
            )

            tokens_generated = len(generated_tokens[0]) - tokens.size(0)
            tokens_sec = tokens_generated / t
            logger.info(
                f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
            logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
