from argparse import ArgumentParser
from pathlib import Path

from ebfm.config import load_yaml, merge_yaml_into_args
from ebfm.modeling_ebfm import resolve_tokenizer_dir
from ebfm.dataset import generate_dataloader
from ebfm.dataset_sft import generate_sft_dataloader
from ebfm.modeling_ebfm import __all__ as model_exports
from ebfm.tokenizer import __all__ as tokenizer_exports


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_explicit_cli_values_override_yaml_values():
    parser = ArgumentParser()
    parser.add_argument("--model-size", default="d6")
    parser.add_argument("--peak-learning-rate", type=float, default=0.0)
    parser.add_argument("--time_embedding", action="store_true", default=False)
    args = parser.parse_args(["--model-size", "d6", "--peak-learning-rate", "0.5"])
    merged = merge_yaml_into_args(
        parser,
        args,
        {"model_size": "d26", "peak_learning_rate": 0.0012, "time_embedding": True},
        ["--model-size", "d6", "--peak-learning-rate", "0.5"],
    )

    assert merged.model_size == "d6"
    assert merged.peak_learning_rate == 0.5
    assert merged.time_embedding is True


def test_yaml_environment_defaults_are_portable(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data_root: '${EBFM_TEST_DATA:-data}'\nasset_root: '${EBFM_TEST_ASSET}'\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("EBFM_TEST_DATA", raising=False)
    monkeypatch.setenv("EBFM_TEST_ASSET", str(tmp_path / "assets"))
    values = load_yaml(config_path)

    assert values == {
        "data_root": "data",
        "asset_root": str(tmp_path / "assets"),
    }


def test_public_boundaries_expose_the_paper_path():
    assert "EBFM_NLP" in model_exports
    assert "EBFMTimeConcat" in model_exports
    assert "get_tokenizer" in tokenizer_exports
    assert "EBFMTokenizerWrapper" in tokenizer_exports
    assert callable(generate_dataloader)
    assert callable(generate_sft_dataloader)


def test_checkpoint_can_discover_adjacent_tokenizer(tmp_path):
    checkpoint = tmp_path / "ebfm.ckpt"
    tokenizer_dir = tmp_path / "tokenizer"
    checkpoint.touch()
    tokenizer_dir.mkdir()

    assert resolve_tokenizer_dir(str(checkpoint)) == tokenizer_dir.resolve()


def test_public_task_boundary_is_flat_and_explicit():
    assert sorted(path.name for path in (REPO_ROOT / "runs").glob("*.sh")) == [
        "chat.sh",
        "eval.sh",
        "prepare_data.sh",
        "qa.sh",
        "sft.sh",
        "train.sh",
    ]
    assert sorted(path.name for path in (REPO_ROOT / "scripts").glob("*.py")) == [
        "__init__.py",
        "chat.py",
        "eval.py",
        "prepare_data.py",
        "qa.py",
        "sft.py",
        "train.py",
    ]
    assert not list((REPO_ROOT / "runs").rglob("*.py"))
    assert "scripts.eval" in (REPO_ROOT / "scripts" / "qa.py").read_text()
    assert "scripts.train" in (REPO_ROOT / "scripts" / "sft.py").read_text()
