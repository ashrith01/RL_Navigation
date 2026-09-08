# Contributing

Thanks for helping improve RL Navigation.

## Workflow

1. Fork the repository and create a focused branch.
2. Keep algorithm changes reproducible by documenting seeds, environment settings, and hyperparameters.
3. Update the README when commands, outputs, or experiment definitions change.
4. Run the checks below before opening a pull request.
5. Explain what changed, why it changed, and which generated results are affected.

## Local checks

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q .
```

For changes to training or evaluation logic, run `python3 main.py` and review the updated files under `outputs/`. Avoid committing virtual environments, caches, or unrelated generated files.

## Pull requests

Keep each pull request limited to one logical change. Include reproduction steps and call out any result differences caused by randomness, configuration changes, or dependency updates.
