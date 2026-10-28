# Validation gates

| Gate | Tool | Pass condition |
|---|---|---|
| 1 | build_oscd_manifest.py | exact 11/3/10 frozen split |
| 2 | verify_oscd.py | overall PASS; 24 cities |
| 3 | audit_oscd_values.py | raw range report generated; anomalies visible |
| 4 | prepare_oscd.py --dry-run | all 24 scenes align; patch counts resolved |
| 5 | prepare_oscd.py | final arrays and metadata written |
| 6 | validate_prepared.py | shape/dtype/labels/SHA256 PASS |
| 7 | stage_kaggle_oscd.py | release folder audited; optional network upload |

For the verified dataset/grid, the project protocol yields 860 train, 100 validation, and 146 test patches. The prior validated project run also reported 417 change-containing and 443 negative train patches, which produces 1,694 sampler indices after 3x positive replication.
