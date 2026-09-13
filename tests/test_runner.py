from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sandbox_runner", REPO / "sandbox" / "runner.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@pytest.fixture(scope="module")
def inputs():
    data_repo = runner.resolve_data_repo()
    runner.assert_pinned_data(data_repo)
    persona, decision, system_prompt = runner.load_inputs(data_repo)
    return persona, decision, system_prompt


def test_pinned_coverage_matches_phase3(inputs):
    persona, _, _ = inputs
    coverage = {
        (row["persona"], row["category"]): row["unique_content"]
        for row in runner.coverage_records(persona)
    }
    assert coverage[("pirate", "A")] == 50
    assert coverage[("poet", "A")] == 50
    assert coverage[("evil", "E")] == 86
    assert coverage[("malicious-manipulative", "F")] == 106
    assert coverage[("assistant", "A")] == 169
    assert coverage[("assistant", "E")] == 175
    assert coverage[("assistant", "F")] >= 50


@pytest.mark.parametrize("condition_id", tuple(runner.CONDITIONS))
def test_all_phase3_conditions_validate(inputs, condition_id):
    persona, decision, _ = inputs
    report = runner.validate_condition(
        persona, decision, runner.CONDITIONS[condition_id]
    )
    assert report["sample_count"] == 20
    if runner.CONDITIONS[condition_id].arm == "baseline":
        assert report["feed_posts"] == 0
    else:
        assert report["feed_posts"] == report["feed_unique"] == 50


def test_choice_shuffle_is_stable_and_target_maps_to_assistant(inputs):
    _, decision, _ = inputs
    condition = runner.CONDITIONS["pirate_a"]
    kwargs = {"seed": runner.CONFIG.seed, "sample_count": runner.CONFIG.sample_count}
    first = runner.build_decisions(decision, condition, **kwargs)
    second = runner.build_decisions(decision, condition, **kwargs)
    assert first == second
    assert {trial["target"] for trial in first} == {"A", "B", "C"}
    for trial in first:
        index = runner.LETTERS.index(trial["target"])
        assert trial["option_order"][index] == "assistant"


def test_parser_has_strict_single_choice_rule():
    assert runner.parse_answer("ANSWER: B", 3) == "B"
    assert runner.parse_answer("b", 3) == "B"
    assert runner.parse_answer("reasoning\nANSWER: C", 3) == "C"
    assert runner.parse_answer("ANSWER: D", 3) is None
    assert runner.parse_answer("A or B", 3) is None
    assert runner.parse_answer("", 3) is None


def test_live_run_requires_confirmation_before_api_or_model_access(tmp_path):
    with pytest.raises(PermissionError, match="--yes"):
        runner.run_live(["a_baseline"], tmp_path, confirmed=False)


def test_generation_kwargs_are_accepted_by_inspect():
    from inspect_ai.model import GenerateConfig

    kwargs = runner.generation_kwargs()
    assert "max_tokens" not in kwargs
    assert runner.CONFIG.feed_max_tokens == 1024
    assert runner.CONFIG.decision_max_tokens == 2048
    GenerateConfig(**kwargs)
