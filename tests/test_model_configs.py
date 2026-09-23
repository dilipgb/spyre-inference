# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Spyre model registry (model_configs.yaml + Python loader)."""

import pytest

from spyre_inference.config import (
    ContinuousBatchingConfig,
    DeviceConfig,
    ModelEntry,
    lookup_config,
    model_registry,
)


def test_registry_loads_without_error():
    reg = model_registry()
    assert isinstance(reg, dict)
    assert len(reg) > 0


def test_every_entry_is_a_model_entry():
    for model_id, entry in model_registry().items():
        assert isinstance(entry, ModelEntry), f"bad type for {model_id}"
        assert entry.model_id == model_id


def test_every_entry_has_at_least_one_cb_config():
    for model_id, entry in model_registry().items():
        assert entry.continuous_batching_configs, (
            f"{model_id} has no continuous_batching_configs"
        )


def test_all_cb_configs_are_typed():
    for model_id, entry in model_registry().items():
        for cfg in entry.continuous_batching_configs:
            assert isinstance(cfg, ContinuousBatchingConfig), (
                f"{model_id}: unexpected type {type(cfg)}"
            )
            assert isinstance(cfg.device_config, DeviceConfig)


@pytest.mark.parametrize(
    "model_id",
    [
        "google/gemma-4-26B-A4B-it",
        "sentence-transformers/all-MiniLM-L6-v2",
        "ibm-granite/granite-embedding-278m-multilingual",
        "FacebookAI/roberta-large-mnli",
        "dslim/bert-base-NER",
        "Jean-Baptiste/roberta-large-ner-english",
    ],
)
def test_known_models_are_present(model_id):
    assert model_id in model_registry(), f"{model_id} missing from registry"


def test_architecture_is_a_dict():
    for model_id, entry in model_registry().items():
        assert isinstance(entry.architecture, dict), (
            f"{model_id}: architecture is {type(entry.architecture)}, expected dict"
        )
        assert entry.architecture, f"{model_id}: architecture dict is empty"


def test_architecture_has_model_type():
    for model_id, entry in model_registry().items():
        arch = entry.architecture
        # model_type may be at top level or inside text_config for VLMs
        has_model_type = "model_type" in arch or (
            isinstance(arch.get("text_config"), dict) and "model_type" in arch["text_config"]
        )
        assert has_model_type, f"{model_id}: no model_type in architecture"


def test_cb_config_positive_fields():
    for model_id, entry in model_registry().items():
        for cfg in entry.continuous_batching_configs:
            assert cfg.tp_size >= 1, f"{model_id}: tp_size < 1"
            assert cfg.max_model_len > 0, f"{model_id}: max_model_len <= 0"
            assert cfg.max_num_seqs > 0, f"{model_id}: max_num_seqs <= 0"


def test_cb_config_tp_size_is_power_of_two():
    for model_id, entry in model_registry().items():
        for cfg in entry.continuous_batching_configs:
            tp = cfg.tp_size
            assert tp & (tp - 1) == 0, f"{model_id}: tp_size={tp} is not a power of two"


def test_device_config_env_vars_are_dict():
    for model_id, entry in model_registry().items():
        for cfg in entry.continuous_batching_configs:
            assert isinstance(cfg.device_config.env_vars, dict), (
                f"{model_id}: device_config.env_vars is not a dict"
            )


def test_num_gpu_blocks_override_is_positive_or_none():
    for model_id, entry in model_registry().items():
        for cfg in entry.continuous_batching_configs:
            override = cfg.device_config.num_gpu_blocks_override
            if override is not None:
                assert override > 0, (
                    f"{model_id}: num_gpu_blocks_override={override} is not positive"
                )


def test_lookup_config_returns_matching_entry():
    cfg = lookup_config("google/gemma-4-26B-A4B-it", tp_size=4, max_model_len=32768)
    assert cfg is not None
    assert cfg.tp_size == 4
    assert cfg.max_model_len == 32768
    assert cfg.device_config.env_vars.get("FLEX_HDMA_P2PSIZE") == 268435456


def test_lookup_config_returns_none_for_missing_model():
    assert lookup_config("not/a-real-model", tp_size=1, max_model_len=4096) is None


def test_lookup_config_returns_none_for_missing_combination():
    # Gemma-4 only has a TP=4 config; TP=1 doesn't exist.
    assert lookup_config("google/gemma-4-26B-A4B-it", tp_size=1, max_model_len=32768) is None


def test_registry_is_cached():
    """model_registry() must return the same dict object on repeated calls."""
    assert model_registry() is model_registry()


