#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$SCRIPT_DIR}"

POLICY_CONFIG_FILE="${ROOT_DIR}/src/openpi/policies/policy_config.py"
MODEL_FILE="${ROOT_DIR}/src/openpi/models/model.py"

python3 - "$POLICY_CONFIG_FILE" "$MODEL_FILE" <<'PY'
import pathlib
import sys

policy_config_file = pathlib.Path(sys.argv[1])
model_file = pathlib.Path(sys.argv[2])


def replace_or_exit(path: pathlib.Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"[skip] {path.name}: {label} already applied")
        return
    if old not in text:
        raise SystemExit(f"[error] Could not find expected block for {label} in {path}")
    path.write_text(text.replace(old, new))
    print(f"[ok] {path.name}: {label}")


replace_or_exit(
    policy_config_file,
    """    # Check if this is a PyTorch model by looking for model.safetensors\n    weight_path = os.path.join(checkpoint_dir, "model.safetensors")\n    is_pytorch = os.path.exists(weight_path)\n""",
    """    # Check if this is a PyTorch model by looking for either a monolithic\n    # safetensors file or a sharded safetensors index.\n    single_weight_path = os.path.join(checkpoint_dir, "model.safetensors")\n    sharded_index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")\n    if os.path.exists(single_weight_path):\n        weight_path = single_weight_path\n    elif os.path.exists(sharded_index_path):\n        weight_path = sharded_index_path\n    else:\n        weight_path = single_weight_path\n    is_pytorch = os.path.exists(single_weight_path) or os.path.exists(sharded_index_path)\n""",
    "detect sharded safetensors checkpoints",
)

replace_or_exit(
    model_file,
    """import abc\nfrom collections.abc import Sequence\nimport dataclasses\nimport enum\nimport logging\nimport pathlib\nfrom typing import Generic, TypeVar\n""",
    """import abc\nfrom collections.abc import Sequence\nimport dataclasses\nimport enum\nimport json\nimport logging\nimport pathlib\nfrom typing import Generic, TypeVar\n""",
    "add json import",
)

replace_or_exit(
    model_file,
    """    def load_pytorch(self, train_config, weight_path: str):\n        logger.info(f"train_config: {train_config}")\n        model = pi0_pytorch.PI0Pytorch(config=train_config.model)\n        safetensors.torch.load_model(model, weight_path)\n        return model\n""",
    """    def load_pytorch(self, train_config, weight_path: str):\n        logger.info(f"train_config: {train_config}")\n        model = pi0_pytorch.PI0Pytorch(config=train_config.model)\n\n        weight_path_obj = pathlib.Path(weight_path)\n        if weight_path_obj.name == "model.safetensors.index.json":\n            if not weight_path_obj.exists():\n                raise FileNotFoundError(f"Sharded index file not found at {weight_path_obj}")\n\n            with weight_path_obj.open("r", encoding="utf-8") as f:\n                index_data = json.load(f)\n\n            weight_map = index_data.get("weight_map", {})\n            if not isinstance(weight_map, dict) or not weight_map:\n                raise ValueError(\n                    f"Invalid sharded checkpoint index at {weight_path_obj}: missing non-empty weight_map"\n                )\n\n            shard_names = sorted(set(weight_map.values()))\n            missing_shards = [\n                shard_name for shard_name in shard_names if not (weight_path_obj.parent / shard_name).exists()\n            ]\n            if missing_shards:\n                raise FileNotFoundError(\n                    f"Missing safetensors shards under {weight_path_obj.parent}: {missing_shards}"\n                )\n\n            state_dict = {}\n            for shard_name in shard_names:\n                shard_path = weight_path_obj.parent / shard_name\n                shard_state = safetensors.torch.load_file(str(shard_path))\n                duplicated_keys = set(state_dict).intersection(shard_state)\n                if duplicated_keys:\n                    raise ValueError(\n                        f"Duplicated tensor keys detected across shards: {sorted(duplicated_keys)[:10]}"\n                    )\n                state_dict.update(shard_state)\n\n            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)\n            if unexpected_keys:\n                raise ValueError(\n                    f"Unexpected checkpoint keys when loading sharded safetensors: {unexpected_keys[:10]}"\n                )\n            if missing_keys:\n                logger.warning(\n                    "Missing model keys when loading sharded safetensors (showing first 10): %s",\n                    missing_keys[:10],\n                )\n        else:\n            safetensors.torch.load_model(model, str(weight_path_obj))\n\n        return model\n""",
    "load sharded safetensors index",
)
PY

echo "Patch applied successfully."
