"""D-104: a non-finite likelihood row stops training by name; the minibatch normalizer is a recorded, opt-in choice."""
import pytest
import torch

from fpp.experiments.train import BATCH_NORMALIZERS, NonFiniteLikelihoodError, batch_loss, run_arm


def test_weight_normalizer_is_the_registered_weighted_mean():
    lp = torch.tensor([-1.0, -2.0, -4.0], dtype=torch.float64)
    w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    out = batch_loss(lp, w, ["a", "b", "c"], "weight")
    assert float(out) == pytest.approx((1 * 1 + 2 * 2 + 3 * 4) / 6)
    assert float(batch_loss(lp, w, ["a", "b", "c"])) == float(out)        # the default is the registered recipe


def test_size_normalizer_divides_by_the_batch_size():
    lp = torch.tensor([-1.0, -2.0, -4.0], dtype=torch.float64)
    w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert float(batch_loss(lp, w, ["a", "b", "c"], "size")) == pytest.approx((1 * 1 + 2 * 2 + 3 * 4) / 3)
    w_mean_one = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)        # weights averaging one: the two agree
    assert torch.isclose(batch_loss(lp, w_mean_one, ["a", "b", "c"], "size"), batch_loss(lp, w_mean_one, ["a", "b", "c"], "weight"))


@pytest.mark.parametrize("bad", [float("nan"), float("-inf")])
def test_non_finite_row_stops_training_and_names_the_unit(bad):
    lp = torch.tensor([-1.0, bad, -3.0], dtype=torch.float64)
    with pytest.raises(NonFiniteLikelihoodError, match=r"1 of 3 .*ids \['u2'\]"):
        batch_loss(lp, torch.ones(3, dtype=torch.float64), ["u1", "u2", "u3"])


def test_unknown_normalizer_is_rejected_before_any_data_is_read(tmp_path):
    assert BATCH_NORMALIZERS == ("weight", "size")
    with pytest.raises(ValueError, match="batch_normalizer"):
        run_arm(tables=tmp_path / "no_tables", fold_season=2023, arm="A", seed=1, out=tmp_path / "out", batch_normalizer="mean")
    with pytest.raises(ValueError):
        batch_loss(torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64), ["a", "b"], "mean")
