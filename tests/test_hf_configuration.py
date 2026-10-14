import pytest


transformers = pytest.importorskip("transformers")

from ebfm.configuration_ebfm import EBFMConfig


def test_ebfm_config_round_trip(tmp_path):
    config = EBFMConfig(model_size="d4", vocab_size=128, context_length=32)
    config.save_pretrained(tmp_path)
    restored = EBFMConfig.from_pretrained(tmp_path)

    assert restored.model_type == "ebfm"
    assert restored.model_size == "d4"
    assert restored.vocab_size == 128
    assert restored.context_length == 32
    assert restored.to_hparams()["mcmc_num_steps"] == 1
