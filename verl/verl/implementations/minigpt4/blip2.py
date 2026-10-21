"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see LICENSE_Lavis.md in this directory or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os

import torch
import torch.nn as nn

from ._common import download_cached_file
from ._common import is_url
from .base_model import BaseModel
from .qformer import BertConfig, BertLMHeadModel
from .eva_vit import create_eva_vit_g
from transformers import BertTokenizer


# Default hub name used when no local BERT path is supplied.  Q-Former only needs
# the BERT *config* and *tokenizer*, never the BERT weights, so this is the plain
# ``bert-base-uncased`` repo id rather than a full checkpoint.
BERT_MODEL_NAME = "bert-base-uncased"


def _resolve_bert_model_path(bert_model_path):
    """Resolve the BERT source used for the Q-Former config and tokenizer.

    ``bert_model_path`` is a *local* directory (typically the HF snapshot dir
    containing ``config.json`` + ``vocab.txt``) and takes precedence over the
    default ``bert-base-uncased`` hub name.  Loading is always offline-only
    (``local_files_only=True``); a missing directory raises immediately instead
    of silently falling back to a network download.
    """
    if bert_model_path:
        path = os.path.expanduser(os.fspath(bert_model_path))
        if not os.path.isdir(path):
            raise FileNotFoundError(
                "BERT model path is not a readable directory: {!r}. Set "
                "bert_model_path to the local bert-base-uncased snapshot "
                "directory (containing config.json + vocab.txt).".format(path)
            )
        return path
    return BERT_MODEL_NAME


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right", bert_model_path=None):
        path = _resolve_bert_model_path(bert_model_path)
        tokenizer = BertTokenizer.from_pretrained(
            path, truncation_side=truncation_side, local_files_only=True
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, bert_model_path=None):
        path = _resolve_bert_model_path(bert_model_path)
        encoder_config = BertConfig.from_pretrained(path, local_files_only=True)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
        cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision,
        vit_pretrained_path=None,
    ):
        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4 or InstructBLIP"
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision,
            pretrained_path=vit_pretrained_path,
        )

        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
