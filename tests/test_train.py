"""End-to-end test of the training script on the tiny built-in corpus (CPU)."""

import json
import os

import pytest

from cs336_basics.data import prepare_sample_data
from cs336_basics.train import main, parse_args


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("smoke_data")
    prepare_sample_data(str(path), vocab_size=300, num_docs=120)
    return str(path)


def argv_for(data_dir, ckpt_dir, *extra):
    return [
        "--smoke",
        "--data-dir", data_dir,
        "--ckpt-dir", str(ckpt_dir),
        "--no-progress",
        "--eval-batches", "2",
        *extra,
    ]


def test_smoke_defaults_apply_but_explicit_flags_win():
    args = parse_args(["--smoke", "--max-iters", "7"])
    assert args.d_model == 64
    assert args.max_iters == 7


def test_smoke_run_decreases_loss_and_writes_checkpoint(data_dir, tmp_path):
    result = main(argv_for(data_dir, tmp_path / "ckpt", "--max-iters", "50"))
    assert result["final_iteration"] == 50
    assert os.path.exists(result["checkpoint"])
    first, last = result["train_losses"][0][1], result["train_losses"][-1][1]
    assert last < first
    assert result["val_losses"]
    with open(tmp_path / "ckpt" / "config.json") as f:
        assert json.load(f)["model_kwargs"]["d_model"] == 64


def test_training_resumes_from_last_checkpoint(data_dir, tmp_path):
    ckpt = tmp_path / "ckpt"
    first = main(argv_for(data_dir, ckpt, "--max-iters", "10", "--eval-interval", "5",
                          "--ckpt-interval", "5", "--log-interval", "5"))
    assert first["final_iteration"] == 10

    second = main(argv_for(data_dir, ckpt, "--max-iters", "20", "--eval-interval", "5",
                           "--ckpt-interval", "5", "--log-interval", "5", "--resume", "auto"))
    assert second["start_iteration"] == 10
    assert second["final_iteration"] == 20
    # Metrics from before the interruption are kept and extended.
    steps = [s for s, _ in second["train_losses"]]
    assert steps[0] == 1 and steps[-1] == 20


def test_resume_auto_without_checkpoint_starts_fresh(data_dir, tmp_path):
    result = main(argv_for(data_dir, tmp_path / "new", "--max-iters", "5", "--resume", "auto"))
    assert result["start_iteration"] == 0


@pytest.mark.parametrize("flags", [["--no-rmsnorm"], ["--post-norm"], ["--no-rope"], ["--ffn-type", "silu"]])
def test_ablation_flags_train(data_dir, tmp_path, flags):
    result = main(argv_for(data_dir, tmp_path / "abl", "--max-iters", "5", *flags))
    assert result["final_iteration"] == 5
