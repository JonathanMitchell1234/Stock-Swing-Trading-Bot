import asyncio
from pathlib import Path

import pytest

import config
from dashboard import server


def test_coerce_config_value_parses_bool_strings():
    assert server._coerce_config_value("HMM_MOMENTUM_OVERRIDE_ENABLED", "false") is False
    assert server._coerce_config_value("HMM_MOMENTUM_OVERRIDE_ENABLED", "0") is False
    assert server._coerce_config_value("HMM_MOMENTUM_OVERRIDE_ENABLED", "true") is True
    assert server._coerce_config_value("HMM_MOMENTUM_OVERRIDE_ENABLED", 1) is True


def test_coerce_config_value_rejects_invalid_bool_string():
    with pytest.raises(ValueError):
        server._coerce_config_value("HMM_MOMENTUM_OVERRIDE_ENABLED", "not-a-bool")


def test_patch_config_applies_false_bool(monkeypatch, tmp_path: Path):
    overrides_path = tmp_path / "config_overrides.json"
    monkeypatch.setattr(server, "CONFIG_OVERRIDES_PATH", overrides_path)
    monkeypatch.setattr(config, "HMM_MOMENTUM_OVERRIDE_ENABLED", True)

    req = server.ConfigPatchRequest(
        updates={"HMM_MOMENTUM_OVERRIDE_ENABLED": "false"}
    )
    result = asyncio.run(server.patch_config(req))

    assert result["ok"] is True
    assert config.HMM_MOMENTUM_OVERRIDE_ENABLED is False
    assert '"HMM_MOMENTUM_OVERRIDE_ENABLED": false' in overrides_path.read_text()