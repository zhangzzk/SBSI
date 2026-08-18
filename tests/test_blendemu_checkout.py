"""BlendEMU is a checkout on the import path, so SBSI must locate it without pip."""

import subprocess
import sys
import textwrap

import pytest

from sbsi import blendemu_checkout
from sbsi.blendemu_checkout import ROOT_ENV_VAR, find_checkout


def _make_checkout(root, body="VALUE = 1\n"):
    package = root / "blendemu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(body)
    return root


def test_env_var_names_the_repository_root(tmp_path, monkeypatch):
    root = _make_checkout(tmp_path / "clone")
    monkeypatch.setenv(ROOT_ENV_VAR, str(root))
    assert find_checkout() == root.resolve()


def test_inner_package_directory_is_also_accepted(tmp_path, monkeypatch):
    root = _make_checkout(tmp_path / "clone")
    monkeypatch.setenv(ROOT_ENV_VAR, str(root / "blendemu"))
    assert find_checkout() == root.resolve()


def test_sibling_of_the_sbsi_checkout_is_the_fallback(tmp_path, monkeypatch):
    root = _make_checkout(tmp_path / "clone")
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(blendemu_checkout, "SIBLING_ROOT", root / "blendemu")
    assert find_checkout() == root.resolve()


def test_directory_without_an_init_is_not_a_checkout(tmp_path, monkeypatch):
    # A bare `blendemu/` directory is what the import system would otherwise pick up as an
    # empty namespace package -- the failure mode when a kernel starts beside a clone.
    (tmp_path / "empty" / "blendemu").mkdir(parents=True)
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path / "empty"))
    monkeypatch.setattr(blendemu_checkout, "SIBLING_ROOT", tmp_path / "absent")
    assert find_checkout() is None


def _run_import(tmp_path, env_root, cwd):
    """Import BlendEMU in a fresh interpreter, since importing mutates sys.path."""

    script = textwrap.dedent(
        f"""
        import os, sys
        os.environ["{ROOT_ENV_VAR}"] = {str(env_root)!r}
        sys.path.insert(0, {str(blendemu_checkout.REPOSITORY_ROOT)!r})
        from sbsi.blendemu_checkout import import_blendemu
        print(import_blendemu().__file__)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script], cwd=cwd, capture_output=True, text=True
    )


def test_import_finds_the_checkout_named_by_the_env_var(tmp_path):
    root = _make_checkout(tmp_path / "clone")
    result = _run_import(tmp_path, root, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(root / "blendemu" / "__init__.py")


def test_import_wins_over_a_namespace_package_in_the_working_directory(tmp_path):
    # Running with the clone's *parent* as the working directory: `blendemu` resolves to
    # the repository directory, which has no __init__.py. The checkout must still win.
    root = _make_checkout(tmp_path / "blendemu")
    result = _run_import(tmp_path, root, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(root / "blendemu" / "__init__.py")


def test_missing_checkout_reports_how_to_point_at_one(tmp_path, monkeypatch):
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(blendemu_checkout, "SIBLING_ROOT", tmp_path / "absent")
    monkeypatch.setattr(blendemu_checkout, "_try_import", lambda: None)
    monkeypatch.delitem(sys.modules, "blendemu", raising=False)
    with pytest.raises(ImportError, match=ROOT_ENV_VAR):
        blendemu_checkout.import_blendemu()


def test_a_missing_blendemu_requirement_names_the_install_that_fixes_it(tmp_path, monkeypatch):
    # Installing SBSI cannot pull in BlendEMU's requirements, so a fresh environment hits
    # this before anything else. A bare "No module named 'xgboost'" is not actionable.
    _make_checkout(tmp_path / "clone", body="import xgboost\n")
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path / "clone"))
    monkeypatch.setitem(sys.modules, "xgboost", None)
    monkeypatch.delitem(sys.modules, "blendemu", raising=False)
    monkeypatch.setattr(blendemu_checkout, "EXTRA_REQUIREMENTS", ("xgboost", "joblib"))
    with pytest.raises(ImportError, match=r"pip install xgboost joblib"):
        blendemu_checkout.import_blendemu()


def test_a_wrong_env_var_fails_instead_of_falling_back(tmp_path, monkeypatch):
    # An explicit setting is authoritative: a typo must not silently resolve to some other
    # copy of BlendEMU in the environment.
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path / "typo"))
    monkeypatch.delitem(sys.modules, "blendemu", raising=False)
    with pytest.raises(ImportError, match="is not a BlendEMU checkout"):
        blendemu_checkout.import_blendemu()
