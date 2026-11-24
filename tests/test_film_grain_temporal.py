import pytest
import graphics.film_grain_temporal as mod

CASES = [
    ("temporal_correlation", (1,)),
    ("strength_by_region", (1,)),
    ("seed_per_shot", (1,)),
    ("grain_before_tonemap", (1,)),
    ("size_by_physical", (1,)),
    ("soft_limit", (1,)),
    ("generate", (1,)),
    ("sample", (1,)),
    ("composite", (1,)),
    ("render", (1,)),
    ("export", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
