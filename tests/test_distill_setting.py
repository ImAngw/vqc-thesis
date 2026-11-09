import importlib.util
import sys
from pathlib import Path

import torch

from classic_heads import MLPHead
from rl_train.policy import Policy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("distill", ROOT / "quantum_heads" / "analysis" / "distill.py")
distill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distill)


def test_teacher_of_the_report_setting_loads_and_reads_5x5_observations(tmp_path):
    """The teachers of the distillation are CNN / window-5 policies (environment v8): the loader must build that one."""
    reference = Policy(MLPHead(hidden=128), **distill.POLICY_KW)
    torch.save(reference.state_dict(), tmp_path / "teacher.pt")
    teacher = distill.load_teacher(str(tmp_path / "teacher.pt"), "mlp128")
    obs = {"perception": torch.zeros(2, 1, 5, 5), "navigator": torch.zeros(2, 17), "memory": torch.zeros(2, 10, 10)}
    logits, _ = teacher(obs)
    assert logits.shape == (2, 8)


def test_environment_setting_is_the_v8_one():
    from rl_train.envs import make_env
    env = make_env(**distill.ENV_KW)
    obs, _ = env.reset(seed=0)
    assert obs["perception"].shape[-2:] == (5, 5)
