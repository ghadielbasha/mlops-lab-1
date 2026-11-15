from src.food11.promote import quality_gate


def test_strong_candidate_is_promoted():
    promoted, required, two_se = quality_gate(0.8, 0.6, 1000)
    assert promoted
    assert required > 0.6
    assert two_se > 0


def test_weak_candidate_is_challenger():
    promoted, required, _ = quality_gate(0.61, 0.6, 1000)
    assert not promoted
    assert required > 0.61


def test_zero_test_size_is_invalid():
    try:
        quality_gate(0.8, 0.6, 0)
    except ValueError:
        return
    raise AssertionError("Zero test images must be rejected")
