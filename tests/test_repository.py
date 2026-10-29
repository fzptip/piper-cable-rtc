"""Offline packaging checks: mock executables, never ROS or model startup."""

import dataclasses
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if BASH is None:
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    if candidate.is_file():
        BASH = str(candidate)


def write(path, content):
    path.write_bytes(content.encode("utf-8"))
    path.chmod(0o755)
    return path.as_posix()


@unittest.skipUnless(BASH, "Bash is required for launcher smoke checks")
class LauncherTests(unittest.TestCase):
    def run_script(self, script, env_file_key, env_text, args):
        with tempfile.TemporaryDirectory(prefix="piper launcher ") as temporary:
            folder = Path(temporary)
            mock_python = write(folder / "mock_python.sh", "#!/usr/bin/env bash\n"
                                'printf "PWD=%s\\n" "$PWD"\n'
                                'printf "PYTHONPATH=%s\\n" "${PYTHONPATH:-}"\n'
                                'printf "CHECKPOINT=%s\\n" "${RTC_CHECKPOINT:-}"\n'
                                'printf "ARG=%s\\n" "$@"\n')
            mock_conda = write(folder / "conda.sh", 'conda() { [[ "$1" == activate && "$2" == offline ]]; }\n')
            mock_ros = write(folder / "ros.sh", 'export MOCK_ROS_SOURCED=yes\n')
            env_file = write(folder / "local.env", env_text.format(
                python=mock_python, conda=mock_conda, ros=mock_ros))
            environment = os.environ.copy()
            for key in list(environment):
                if key.startswith(("PIPER_", "RTC_")):
                    environment.pop(key)
            environment[env_file_key] = env_file
            result = subprocess.run([BASH, "--noprofile", "--norc", str(script), *args],
                                    cwd=folder, env=environment, capture_output=True,
                                    text=True, timeout=20, check=True)
            return result.stdout.splitlines()

    def test_robot_launches_from_unrelated_directory_with_local_environment(self):
        lines = self.run_script(
            ROOT / "robot_client/pi0.5_rtc_control_v3.sh", "PIPER_ENV_FILE",
            'PIPER_CONDA_SH="{conda}"\nPIPER_CONDA_ENV=offline\n'
            'PIPER_ROS_SETUP="{ros}"\nPIPER_PYTHON="{python}"\n',
            ["--publish-rate", "100", "--diagnostics-label", "offline smoke"])
        self.assertTrue(lines[0].replace("\\", "/").endswith("/robot_client"))
        args = [line[4:] for line in lines if line.startswith("ARG=")]
        self.assertTrue(args[0].replace("\\", "/").endswith("/robot_client/pi05_ws_client.py"))
        self.assertEqual(args[-4:], ["--publish-rate", "100", "--diagnostics-label", "offline smoke"])
        for flag, value in (("--action-rate", "30"), ("--publish-rate", "90"),
                            ("--action-interpolation", "linear"), ("--chunk-steps", "15"),
                            ("--rtc-transition-steps", "6"), ("--rtc-smoothing-alpha", "1.0")):
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_server_launches_from_unrelated_directory_with_checkout_sources(self):
        lines = self.run_script(
            ROOT / "server/openpi/deployment/start_rtc.sh", "RTC_ENV_FILE",
            'export RTC_PYTHON="{python}"\nexport RTC_CHECKPOINT="/offline/weights"\n',
            ["--host", "127.0.0.1", "--port", "8000"])
        self.assertTrue(lines[0].replace("\\", "/").endswith("/server/openpi"))
        self.assertIn("CHECKPOINT=/offline/weights", lines)
        python_path = next(line for line in lines if line.startswith("PYTHONPATH="))
        self.assertIn("/server/openpi/src", python_path.replace("\\", "/"))
        args = [line[4:] for line in lines if line.startswith("ARG=")]
        self.assertTrue(args[0].replace("\\", "/").endswith("/scripts/serve_rtc.py"))
        self.assertEqual(args[-4:], ["--host", "127.0.0.1", "--port", "8000"])


def mocked_service():
    """Import the actual entry point with model/GPU modules replaced by mocks."""
    importlib.import_module("numpy")
    factory = Mock()
    factory.create.return_value = types.SimpleNamespace(norm_stats={"actions": "offline"})

    @dataclasses.dataclass(frozen=True)
    class TrainingConfig:
        assets_base_dir: str = "/original/assets"
        name: str = "pi05_cobot_magic_plug_cable_finetune"
        model: str = "offline-model"

        @property
        def data(self):
            return factory

        @property
        def assets_dirs(self):
            return Path(self.assets_base_dir) / self.name

    original = TrainingConfig()
    modules = {name: types.ModuleType(name) for name in (
        "openpi", "openpi.policies", "openpi.policies.policy_config",
        "openpi.policies.rtc_policy", "openpi.serving",
        "openpi.serving.websocket_policy_server", "openpi.training", "openpi.training.config")}
    modules["openpi.training.config"].get_config = Mock(return_value=original)
    modules["openpi.training.config"].__file__ = "offline-training-config.py"
    modules["openpi.policies.policy_config"].create_trained_policy = Mock(return_value="base")
    modules["openpi.policies.rtc_policy"].RTCPolicy = Mock(return_value="wrapped")
    modules["openpi.serving.websocket_policy_server"].WebsocketPolicyServer = Mock(
        side_effect=AssertionError("Offline check must not start a server"))
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("offline_serve_rtc", ROOT / "server/openpi/scripts/serve_rtc.py")
        service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service)
    return service, original


class AssetConfigurationTests(unittest.TestCase):
    def test_environment_paths_reach_policy_loader_without_mutating_training_config(self):
        service, original = mocked_service()
        with patch.dict(os.environ, {"RTC_CHECKPOINT": "/env/weights", "RTC_ASSETS_BASE_DIR": "/env/assets"}):
            self.assertEqual(service.load_policy(), "wrapped")
        call = service.policy_config.create_trained_policy.call_args
        self.assertEqual(call.args[1], "/env/weights")
        self.assertEqual(call.args[0].assets_base_dir, "/env/assets")
        self.assertEqual(original.assets_base_dir, "/original/assets")

    def test_explicit_paths_override_environment(self):
        service, _ = mocked_service()
        with patch.dict(os.environ, {"RTC_CHECKPOINT": "/env/weights", "RTC_ASSETS_BASE_DIR": "/env/assets"}):
            service.load_policy("/explicit/weights", assets_base_dir="/explicit/assets")
        call = service.policy_config.create_trained_policy.call_args
        self.assertEqual(call.args[1], "/explicit/weights")
        self.assertEqual(call.args[0].assets_base_dir, "/explicit/assets")

    def test_missing_statistics_fail_before_loading_model(self):
        service, original = mocked_service()
        original.data.create.return_value.norm_stats = None
        with self.assertRaises(FileNotFoundError):
            service.load_policy("/offline/weights")
        service.policy_config.create_trained_policy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
