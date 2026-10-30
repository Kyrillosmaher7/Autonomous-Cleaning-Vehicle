# Autonomous Cleaning Vehicle

An autonomous cleaning vehicle simulation using Reinforcement Learning.

## Project Goal

Develop an autonomous cleaning agent that learns how to navigate an environment,
clean dirt, avoid obstacles, manage battery usage, and eventually handle
more complex environments.

## Main Technologies

- Python
- Reinforcement Learning
- Q-Learning
- SARSA
- pytest
- GitHub Actions

## Project Structure

```text
src/                Main source code
tests/              Unit and integration tests
configs/            All numerical settings (YAML), see "Configuration"
run.py              Start the visualizer (control panel)
experiments/        Experiment results
notebooks/          Exploration and analysis
docs/               Project documentation
data/               Dataset and generated data
models/             Trained models
reports/            Figures and experiment reports
.github/workflows/  CI/CD workflows
```

## Running the visualizer

```bash
python run.py
```

The control panel opens first. Pick a **version** and a **model**, check
the configuration shown, and press **Run simulation**. Models that need
training (Q-Learning) are trained first, then the simulation window opens.
Closing it returns you to the control panel.

```bash
python run.py --list                            # what is available
python run.py --version v2 --model q_learning   # skip the control panel
python -m cleaning_vehicle.experiments.run_experiment --version v1 --model greedy   # no GUI, prints metrics
```

(`tkinter` is required for the GUI; on Debian/Ubuntu: `sudo apt install python3-tk`.)

## Configuration

Every number lives in `configs/`; nothing is hard-coded in the code.

```text
configs/default.yaml            seed, training/evaluation episodes, window sizes, colors, speeds
configs/environments/*.yaml     one file per VERSION: grid, battery, dirt, obstacles, vision, rewards
configs/agents/*.yaml           one file per MODEL: hyperparameters, whether it needs training
```

An environment file is merged on top of `default.yaml`, so it only needs to
state what differs. Unknown keys and out-of-range values are rejected with a
message naming the setting.

**Add a version or a model:** drop a new YAML file in the matching folder
(with a unique `version:` / `algorithm:` id) and it appears in the control
panel. Set `status: planned` to hide an unfinished entry.

## Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

