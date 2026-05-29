import torch

from lagunavision.train.batching import cosine_warmup_multiplier, pad_visual_batch


def test_pad_visual_batch_right_pads_and_masks() -> None:
    hidden = 3
    seq_a = torch.ones(2, hidden)
    seq_b = torch.ones(5, hidden) * 2
    labels_a = torch.tensor([-100, 7])
    labels_b = torch.tensor([-100, -100, 1, 2, 3])

    batch = pad_visual_batch([seq_a, seq_b], [labels_a, labels_b])

    assert batch["inputs_embeds"].shape == (2, 5, hidden)
    assert batch["attention_mask"].tolist() == [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]
    # padded rows carry -100 so they never contribute to the loss
    assert batch["labels"][0].tolist() == [-100, 7, -100, -100, -100]
    assert batch["labels"][1].tolist() == [-100, -100, 1, 2, 3]
    # padded embedding rows are zeroed
    assert torch.equal(batch["inputs_embeds"][0, 2:], torch.zeros(3, hidden))


def test_cosine_warmup_multiplier_shape() -> None:
    warmup, total = 10, 100
    assert cosine_warmup_multiplier(0, warmup, total) == 0.1
    assert cosine_warmup_multiplier(9, warmup, total) == 1.0
    # cosine peak right after warmup, decaying to min_ratio at the end
    assert cosine_warmup_multiplier(10, warmup, total) == 1.0
    assert abs(cosine_warmup_multiplier(total, warmup, total) - 0.1) < 1e-6
    mid = cosine_warmup_multiplier(55, warmup, total)
    assert 0.1 < mid < 1.0


def test_cosine_warmup_handles_zero_warmup() -> None:
    assert cosine_warmup_multiplier(0, 0, 10) == 1.0
