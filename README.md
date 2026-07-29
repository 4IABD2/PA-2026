# GT CARLA — Autonomous Driving in CARLA

Annual project (4IABD2). A vehicle drives autonomously in the **CARLA** simulator
using computer vision only (a single front RGB camera). Perception modules turn
the camera image into a compact state, and a reinforcement-learning policy (PPO)
outputs the driving commands.

Final model: `v23` — **7/8 on the benchmark, 0 collision, no ground truth**.

---

## Requirements

- **Python 3.10–3.12**
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **CARLA 0.9.16** running on a reachable host (see below)
- An NVIDIA GPU is recommended (for CARLA and the perception models)

## Installation

```bash
uv sync
```

This creates the virtual environment and installs every dependency. Python 3.10
is downloaded automatically if it is not on your system.

## Running CARLA

CARLA 0.9.16 must be running before you launch anything. Start it on any machine
(Docker or native) and note its IP address. Example with Docker:

```bash
docker run -d --gpus all -p 2000-2002:2000-2002 \
    carlasim/carla:0.9.16 \
    ./CarlaUE4.sh -RenderOffScreen -world-port=2000
```

The scripts default to `localhost:2000`. If CARLA runs on another machine, pass
its address with `--host`.

## Usage

All commands run through `uv run` (no need to activate the venv).

### Train the policy

```bash
uv run python scripts/run_rl_training.py --host <CARLA_IP> --timesteps 110000 --tag ppo
```

Artifacts are written to `runs/<date>_<tag>/`: checkpoints, reward curve,
`best_model.zip`, and an automatic 13-scenario evaluation (videos + JSON).

Useful options:

| Option | Default | Description |
|---|---|---|
| `--timesteps N` | 500000 | training length |
| `--host IP` | localhost | CARLA server address |
| `--npcs N` | 18 | NPC vehicles spawned |
| `--pedestrians N` | 6 | NPC pedestrians spawned |
| `--goal-bearing` | off | add the bearing-to-goal input (used by the final model) |

For a quick smoke test, use `--timesteps 1000 --tag smoke`.

### Guided launcher (first run)

Checks CARLA connectivity, the active map and the YOLO weights before training:

```bash
uv run python launch_training.py --host <CARLA_IP>
```

### Integrated demo (remote inference)

Runs the car in CARLA while the policy is served remotely in ONNX by the C++
server in `web/` (start that server first):

```bash
uv run python main.py --host <CARLA_IP>
```

### Export a model to ONNX

```bash
uv run python scripts/export_model_to_onnx.py runs/<run>/best_model.zip
```

## Project structure

```
src/ai/             central decision AI: PPO policy, reward, evaluation
src/perception/     object detection (YOLO11s) + depth (Depth Anything v2)
src/lane_detection/ lane detection (YOLOPv2)
src/navigation/     route planning (A*)
src/dataset/        dataset generation from CARLA
scripts/            training pipeline, ONNX export
web/                C++ ONNX inference server + Python client
main.py             integrated demo (remote inference)
runs/               training artifacts (gitignored)
```

## Formatting

```bash
uv run black .
```
