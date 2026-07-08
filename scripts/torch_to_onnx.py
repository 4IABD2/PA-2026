import sys
from pathlib import Path

import torch

from src.ai.inference.rl_demo import load_model


class OnnxActorWrapper(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Extract features in an export-friendly way
        features = self.policy.extract_features(obs)
        if isinstance(features, tuple):
            features = features[0]

        latent_pi, _ = self.policy.mlp_extractor(features)
        mean_actions = self.policy.action_net(latent_pi)

        # If the policy uses action squashing, keep the output in action range.
        # For your env this is steer/throttle/brake = 3 floats.
        return torch.tanh(mean_actions)


def main(model_path: str, output_path: str | None = None) -> None:
    model = load_model(model_path)

    policy = model.policy.cpu().eval()
    wrapper = OnnxActorWrapper(policy).cpu().eval()

    dummy_obs = torch.zeros((1, 12), dtype=torch.float32)

    if output_path is None:
        output_path = str(Path(model_path).with_suffix(".onnx"))

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_obs,
            output_path,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
            opset_version=17,
        )

    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts/torch_to_onnx.py <model_path> [output_path]"
        )
    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(model_path, output_path)
