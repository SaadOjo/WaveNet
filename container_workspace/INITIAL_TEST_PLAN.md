# Initial Test Plan

Initial dataset subset for early testing and baseline comparisons:

- 10
- 11
- 12
- 13

These correspond to IC1 with:
- balanced full
- imbalanced full
- balanced max5000
- balanced max2500

Usage example:

```bash
python scripts/random_forest.py --target-ids 10 11 12 13
python scripts/support_vector_machine.py --target-ids 10 11 12 13
python scripts/xgboost.py --target-ids 10 11 12 13
```
