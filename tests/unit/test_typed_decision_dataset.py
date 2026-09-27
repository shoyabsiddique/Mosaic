"""Regression test for the round-robin sampling bug: a naive prefix-truncation over
files sorted by domain name silently returned only one domain's examples for any small
max_examples cap (caught via a Phase 3 smoke test's routing diagnostic).
"""
import json

from decision_engine.training.dataset import TypedDecisionDataset, train_val_split


def _write_domain_file(tmp_path, domain: str, n: int):
    d = tmp_path / domain
    d.mkdir()
    with open(d / "data.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"domain": domain, "idx": i}) + "\n")


def test_no_cap_reads_every_example_from_every_domain(tmp_path):
    _write_domain_file(tmp_path, "alpha-domain", 3)
    _write_domain_file(tmp_path, "beta-domain", 3)

    ds = TypedDecisionDataset(tmp_path)
    assert len(ds) == 6
    domains = {ds[i]["domain"] for i in range(len(ds))}
    assert domains == {"alpha-domain", "beta-domain"}


def test_small_cap_still_samples_every_domain_not_just_the_alphabetically_first(tmp_path):
    # "alpha-domain" sorts before "beta-domain" and before "zeta-domain" -- a naive
    # prefix truncation at max_examples=3 would return only alpha-domain's examples.
    _write_domain_file(tmp_path, "alpha-domain", 10)
    _write_domain_file(tmp_path, "beta-domain", 10)
    _write_domain_file(tmp_path, "zeta-domain", 10)

    ds = TypedDecisionDataset(tmp_path, max_examples=6)
    assert len(ds) == 6
    domains = {ds[i]["domain"] for i in range(len(ds))}
    assert domains == {"alpha-domain", "beta-domain", "zeta-domain"}


def test_cap_smaller_than_domain_count_still_does_not_crash(tmp_path):
    _write_domain_file(tmp_path, "alpha-domain", 5)
    _write_domain_file(tmp_path, "beta-domain", 5)

    ds = TypedDecisionDataset(tmp_path, max_examples=1)
    assert len(ds) == 1


def test_cap_larger_than_total_available_returns_everything(tmp_path):
    _write_domain_file(tmp_path, "alpha-domain", 2)
    _write_domain_file(tmp_path, "beta-domain", 2)

    ds = TypedDecisionDataset(tmp_path, max_examples=1000)
    assert len(ds) == 4


def test_train_val_split_covers_every_example_with_no_overlap(tmp_path):
    _write_domain_file(tmp_path, "alpha-domain", 20)
    ds = TypedDecisionDataset(tmp_path)

    train, val = train_val_split(ds, val_fraction=0.25, seed=0)
    assert len(train) + len(val) == 20
    train_ids = {ex["idx"] for ex in train.examples}
    val_ids = {ex["idx"] for ex in val.examples}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(range(20))


def test_train_val_split_is_reproducible_given_same_seed(tmp_path):
    _write_domain_file(tmp_path, "alpha-domain", 20)
    ds = TypedDecisionDataset(tmp_path)

    train_a, val_a = train_val_split(ds, seed=7)
    train_b, val_b = train_val_split(ds, seed=7)
    assert [e["idx"] for e in val_a.examples] == [e["idx"] for e in val_b.examples]


def test_uneven_domain_sizes_do_not_break_round_robin(tmp_path):
    _write_domain_file(tmp_path, "alpha-domain", 1)
    _write_domain_file(tmp_path, "beta-domain", 10)

    ds = TypedDecisionDataset(tmp_path, max_examples=5)
    assert len(ds) == 5
    domains = [ds[i]["domain"] for i in range(len(ds))]
    assert domains.count("alpha-domain") == 1  # exhausted after row 0, rest backfilled from beta
    assert domains.count("beta-domain") == 4
