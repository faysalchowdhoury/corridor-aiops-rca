from __future__ import annotations

import pytest

from corridor_aiops.data.audit import METRIC_NAMES, _capabilities


def test_veremi_gate_allows_only_labeled_detection_emitter_localization_and_scoped_mttd() -> None:
    capabilities = _capabilities("veremi")
    evaluable = {
        metric for metric, capability in capabilities.items() if capability["status"] == "EVALUABLE"
    }

    assert evaluable == {"detection", "emitter_or_node_localization", "mttd"}
    assert capabilities["causal_rca"]["status"] == "NOT_EVALUABLE"


@pytest.mark.parametrize(
    "dataset",
    ["pems_bay", "tampa_cv", "caltrans_lcs", "cisa_kev", "mitre_ics"],
)
def test_unlabeled_or_unaligned_datasets_fail_closed(dataset: str) -> None:
    capabilities = _capabilities(dataset)

    assert set(capabilities) == set(METRIC_NAMES)
    assert {item["status"] for item in capabilities.values()} == {"NOT_EVALUABLE"}


def test_unknown_dataset_has_no_implicit_eligibility() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        _capabilities("unknown")
