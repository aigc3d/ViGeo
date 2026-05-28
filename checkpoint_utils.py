from pathlib import Path

import torch


def checkpoint_path(checkpoint_dir, *parts):
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}. Please place local checkpoints under {checkpoint_dir}."
        )
    return path


def load_checkpoint(checkpoint_dir, *parts, map_location='cpu'):
    return torch.load(checkpoint_path(checkpoint_dir, *parts), map_location=map_location)


def hf_from_pretrained(load_fn, model_id, **kwargs):
    try:
        return load_fn(model_id, local_files_only=True, **kwargs)
    except Exception:
        try:
            return load_fn(model_id, local_files_only=False, **kwargs)
        except Exception as download_error:
            raise RuntimeError(
                f"Failed to load HuggingFace model '{model_id}' from the default local cache, "
                "and automatic download also failed."
            ) from download_error
