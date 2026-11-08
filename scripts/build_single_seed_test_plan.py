#!/usr/bin/env python
"""Create a test-first campaign using one fixed seed per neural method/view."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "main_campaign.json",
    "temporal_campaign.json",
    "mencp_campaign.json",
    "repeatmixer_campaign.json",
    "tgat_campaign.json",
    "dygformer_campaign.json",
    "idgp_campaign.json",
]

jobs = []
for filename in SOURCES:
    plan = json.loads((ROOT / "configs" / filename).read_text())
    for source in plan["jobs"]:
        argv = source["argv"]
        if "--seed" not in argv or str(argv[argv.index("--seed") + 1]) != "42":
            continue
        job = dict(source)
        job["id"] = "test_" + job["id"]
        job["argv"] = list(argv)
        if "--evaluate-test" not in job["argv"]:
            job["argv"].append("--evaluate-test")
        job["output"] = job["output"].replace("/validation.json", "/metrics.json")
        jobs.append(job)

output = {
    "name": "test_priority_single_seed_v2",
    "policy": {
        "seed": 42,
        "priority": "test",
        "validation_repeats": False,
        "note": "Replay existing checkpoints; train missing cells once, then evaluate validation and test.",
    },
    "jobs": jobs,
}
(ROOT / "configs" / "test_priority_single_seed.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2) + "\n"
)
for suffix, methods in {
    "sampling_fast_a": {"TGAT"},
    "sampling_fast_b": {"DyGFormer"},
    "idgp_fast": {"IDGP"},
}.items():
    selected = [
        job for job in jobs
        if "--method" in job["argv"]
        and job["argv"][job["argv"].index("--method") + 1] in methods
    ]
    parallel = dict(output, name="test_priority_" + suffix, jobs=selected)
    (ROOT / "configs" / ("test_priority_" + suffix + ".json")).write_text(
        json.dumps(parallel, ensure_ascii=False, indent=2) + "\n"
    )
for country in ("CN", "US"):
    selected = [
        job for job in jobs
        if "--method" in job["argv"]
        and job["argv"][job["argv"].index("--method") + 1] == "RepeatMixer"
        and job["argv"][job["argv"].index("--dataset") + 1].startswith(country + "_")
    ]
    suffix = "repeat_" + country.lower()
    parallel = dict(output, name="test_priority_" + suffix, jobs=selected)
    (ROOT / "configs" / ("test_priority_" + suffix + ".json")).write_text(
        json.dumps(parallel, ensure_ascii=False, indent=2) + "\n"
    )
print(f"wrote {len(jobs)} jobs")
