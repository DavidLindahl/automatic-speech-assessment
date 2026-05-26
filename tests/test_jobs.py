"""Validate HPC job scripts locally before queue submission.

The checks in this module are intentionally static/lightweight:
- Shell syntax for every script in ``jobs/``.
- Extraction of Python command invocations from scripts.
- Typer argument parsing for ``src/asa/*.py`` entrypoints without invoking them.
- Existence checks for critical path arguments (datasets/configs).
- Optional Hugging Face model ID validation via ``AutoConfig``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest
import typer
from click import Command
from transformers import AutoConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_ROOT = PROJECT_ROOT / "jobs"

EXISTING_FILE_OPTIONS = {
    "--deepspeed",
    "--json-path",
    "--input-json",
    "--dataset-path",
}
EXISTING_DIR_OPTIONS = {"--data-root"}
MODEL_OPTIONS = {"--model-id", "--ref-model-id", "--model-path"}

LOCAL_MODEL_PREFIXES = {"configs", "data", "jobs", "models", "results", "src", "tests"}

VALIDATE_REMOTE_MODELS = os.getenv("ASA_VALIDATE_REMOTE_MODELS", "1") == "1"
STRICT_LOCAL_MODEL_PATHS = os.getenv("ASA_STRICT_LOCAL_MODEL_PATHS", "0") == "1"
STRICT_GENERATED_INPUTS = os.getenv("ASA_STRICT_GENERATED_INPUTS", "0") == "1"

REMOTE_VALIDATION_SKIP_REASON: str | None = None


@dataclass(frozen=True)
class JobInvocation:
    """A parsed Python invocation extracted from a job script.

    Attributes:
        job_script: Path to the job script that contains the command.
        line_number: Starting line number of the logical command.
        runner: High-level command runner used (`torchrun`, `uv`, `python`).
        target: Python file path token (for example ``scripts/eval/evaluate.py``).
        args: Argument tokens passed to the Python target.
        raw: Raw logical command line after continuation-join.
    """

    job_script: Path
    line_number: int
    runner: str
    target: str
    args: tuple[str, ...]
    raw: str


def _list_job_scripts() -> list[Path]:
    """Return every live `.sh` job script under ``jobs/``.

    Skips ``jobs/_archive/`` (historical scripts preserved for reference; they
    may reference modules that have since been removed) and any non-``.sh``
    files such as ``_lib/README.md`` or ``_lib/templates/*.tpl``.
    """
    return sorted(
        path
        for path in JOBS_ROOT.rglob("*.sh")
        if path.is_file() and "_archive" not in path.parts
    )


def _strip_inline_comments(line: str) -> str:
    """Remove shell comments while respecting basic quote boundaries.

    Args:
        line: Raw shell script line.

    Returns:
        Line with trailing comment removed if present outside quotes.
    """
    in_single = False
    in_double = False

    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_script_variables(
    script_text: str,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse simple scalar and array assignments from a shell script.

    Args:
        script_text: Full shell script content.

    Returns:
        Tuple ``(scalars, arrays)`` where scalar values are strings and arrays
        are tokenized lists.
    """
    scalars: dict[str, str] = {}
    arrays: dict[str, list[str]] = {}

    lines = script_text.splitlines()
    index = 0
    while index < len(lines):
        raw = _strip_inline_comments(lines[index]).strip()
        if not raw:
            index += 1
            continue

        array_match = re.match(r"^(?:export\s+)?([A-Za-z_]\w*)=\((.*)$", raw)
        if array_match:
            name = array_match.group(1)
            buffer = [array_match.group(2)]
            while ")" not in buffer[-1] and index + 1 < len(lines):
                index += 1
                buffer.append(_strip_inline_comments(lines[index]).strip())

            joined = "\n".join(buffer)
            if ")" in joined:
                inner = joined.split(")", maxsplit=1)[0]
                try:
                    arrays[name] = shlex.split(inner, posix=True)
                except ValueError:
                    # Array contents contain an unbalanced quote spanning
                    # constructs we don't model. Skip recording rather
                    # than crashing collection.
                    pass

            index += 1
            continue

        scalar_match = re.match(r"^(?:export\s+)?([A-Za-z_]\w*)=(.+)$", raw)
        if scalar_match:
            name = scalar_match.group(1)
            value_raw = scalar_match.group(2).strip()
            try:
                value_tokens = shlex.split(value_raw, posix=True)
            except ValueError:
                # Right-hand side spans multiple lines (heredoc, multi-line
                # $(...), or unclosed quote). We can't expand the scalar
                # from a single line; skip recording it. Downstream token
                # expansion will fall back to leaving ${NAME} literal.
                value_tokens = []
            if len(value_tokens) == 1:
                scalars[name] = value_tokens[0]

        index += 1

    return scalars, arrays


def _iter_logical_lines(script_text: str) -> list[tuple[int, str]]:
    """Join line continuations (``\\``) into logical command lines.

    Args:
        script_text: Full shell script content.

    Returns:
        Logical lines as ``(start_line_number, text)``.
    """
    logical: list[tuple[int, str]] = []
    buffer = ""
    start_line = 1

    for line_number, raw_line in enumerate(script_text.splitlines(), start=1):
        clean = _strip_inline_comments(raw_line).rstrip()
        if not clean.strip():
            continue

        if not buffer:
            start_line = line_number

        if clean.endswith("\\"):
            buffer += clean[:-1] + " "
            continue

        buffer += clean
        logical.append((start_line, buffer.strip()))
        buffer = ""

    if buffer:
        logical.append((start_line, buffer.strip()))

    return logical


def _expand_token(
    token: str, scalars: dict[str, str], arrays: dict[str, list[str]]
) -> str:
    """Expand simple shell variables used by current job scripts.

    Args:
        token: Token to expand.
        scalars: Parsed scalar variables.
        arrays: Parsed array variables.

    Returns:
        Expanded token when a known pattern is matched; original token otherwise.
    """
    array_ref = re.fullmatch(r"\$\{([A-Za-z_]\w*)\[(\d+)\]\}", token)
    if array_ref:
        name = array_ref.group(1)
        idx = int(array_ref.group(2))
        values = arrays.get(name, [])
        if 0 <= idx < len(values):
            return values[idx]
        return token

    braced = re.fullmatch(r"\$\{([A-Za-z_]\w*)\}", token)
    if braced:
        return scalars.get(braced.group(1), token)

    bare = re.fullmatch(r"\$([A-Za-z_]\w*)", token)
    if bare:
        return scalars.get(bare.group(1), token)

    return token


def _expand_tokens(
    tokens: list[str], scalars: dict[str, str], arrays: dict[str, list[str]]
) -> list[str]:
    """Expand simple variable references in a token list."""
    return [_expand_token(token, scalars, arrays) for token in tokens]


def _find_python_target_index(tokens: list[str]) -> int | None:
    """Return index of first Python file token in a command token list."""
    for index, token in enumerate(tokens):
        if token.endswith(".py") and not token.startswith("-"):
            return index
    return None


def _extract_invocations(job_script: Path) -> list[JobInvocation]:
    """Extract Python invocations from a single job script.

    Args:
        job_script: Job script path.

    Returns:
        Parsed invocation records for lines that run Python files.
    """
    script_text = job_script.read_text(encoding="utf-8")
    scalars, arrays = _parse_script_variables(script_text)
    logical_lines = _iter_logical_lines(script_text)

    invocations: list[JobInvocation] = []
    for line_number, command_line in logical_lines:
        try:
            tokens = shlex.split(command_line, posix=True)
        except ValueError:
            continue

        if not tokens:
            continue

        expanded = _expand_tokens(tokens, scalars, arrays)
        runner = expanded[0]
        target_index: int | None = None

        if runner == "torchrun":
            target_index = _find_python_target_index(expanded)
        elif (
            len(expanded) >= 4
            and expanded[0] == "uv"
            and expanded[1] == "run"
            and expanded[2] == "python"
        ):
            target_index = 3 if expanded[3].endswith(".py") else None
            runner = "uv"
        elif runner in {"python", "python3"}:
            target_index = (
                1 if len(expanded) > 1 and expanded[1].endswith(".py") else None
            )

        if target_index is None:
            continue

        target = expanded[target_index]
        args = tuple(expanded[target_index + 1 :])
        invocations.append(
            JobInvocation(
                job_script=job_script,
                line_number=line_number,
                runner=runner,
                target=target,
                args=args,
                raw=command_line,
            )
        )

    return invocations


def _collect_job_invocations() -> list[JobInvocation]:
    """Collect invocations from all job scripts."""
    all_invocations: list[JobInvocation] = []
    for job_script in _list_job_scripts():
        all_invocations.extend(_extract_invocations(job_script))
    return all_invocations


@lru_cache(maxsize=32)
def _load_module_from_path(relative_path: str) -> ModuleType:
    """Load a Python module from a repository-relative file path.

    Args:
        relative_path: Path relative to project root.

    Returns:
        Imported module object.
    """
    module_path = PROJECT_ROOT / relative_path
    module_name = "job_validation_" + re.sub(r"\W+", "_", relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {relative_path}")

    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection during import (typer
    # wraps decorated functions and uses dataclasses internally) can resolve
    # ``sys.modules[cls.__module__]``.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=32)
def _load_typer_command(relative_path: str) -> Command:
    """Load a Click command from a Typer app defined in a script.

    Supports two patterns:
      1. ``app = typer.Typer()`` at module top-level, used with ``app.command``.
      2. ``def main(...): ...`` invoked via ``typer.run(main)`` in ``__main__``.

    Args:
        relative_path: Script path relative to project root.

    Returns:
        Click command generated by Typer.
    """
    module = _load_module_from_path(relative_path)
    app = getattr(module, "app", None)
    if isinstance(app, typer.Typer):
        return typer.main.get_command(app)

    main_fn = getattr(module, "main", None)
    if callable(main_fn):
        wrapper = typer.Typer()
        wrapper.command()(main_fn)
        return typer.main.get_command(wrapper)

    raise RuntimeError(
        f"{relative_path} exposes neither a Typer `app` object nor a `main` "
        f"function suitable for typer.run()."
    )


def _resolve_repo_path(path_value: str) -> Path:
    """Resolve a possibly-relative path against repository root.

    BSUB preambles set ``PROJECT_DIR``/``EXPERIMENT_DIR``/``REPO_ROOT`` to
    the project root at runtime, so paths anchored at those variables are
    repo-root-relative for our static-existence checks. Strip those known
    prefixes before treating the remainder as relative.

    Args:
        path_value: Raw path string from script arguments.

    Returns:
        Absolute path.
    """
    expanded = path_value
    for var in ("$EXPERIMENT_DIR", "${EXPERIMENT_DIR}", "$PROJECT_DIR",
                "${PROJECT_DIR}", "$REPO_ROOT", "${REPO_ROOT}"):
        if expanded.startswith(var):
            expanded = expanded[len(var):].lstrip("/")
            break

    candidate = Path(expanded).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _iter_option_values(args: tuple[str, ...]) -> list[tuple[str, str]]:
    """Extract ``(--flag, value)`` pairs from CLI args.

    Args:
        args: CLI argument tokens.

    Returns:
        Extracted pairs for long options that have attached values.
    """
    pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            index += 1
            continue

        if "=" in token:
            name, value = token.split("=", maxsplit=1)
            pairs.append((name, value))
            index += 1
            continue

        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            pairs.append((token, args[index + 1]))
            index += 2
            continue

        index += 1

    return pairs


def _looks_like_local_model_reference(value: str) -> bool:
    """Heuristically classify a model reference as local-path-like.

    Values starting with BSUB-preamble variables (``$EXPERIMENT_DIR``,
    ``$PROJECT_DIR``, ``$REPO_ROOT``) are treated as local because those
    expand to the project root at runtime.

    Args:
        value: Raw model reference value.

    Returns:
        ``True`` when value appears to point to local artifacts.
    """
    if value.startswith(("/", "./", "../", "~")):
        return True

    if value.startswith(("$EXPERIMENT_DIR", "${EXPERIMENT_DIR}",
                         "$PROJECT_DIR", "${PROJECT_DIR}",
                         "$REPO_ROOT", "${REPO_ROOT}")):
        return True

    first_segment = value.split("/", maxsplit=1)[0]
    return first_segment in LOCAL_MODEL_PREFIXES


def _looks_like_generated_input_artifact(value: str) -> bool:
    """Return whether a path looks like a generated training artifact.

    Args:
        value: Raw path option value from a job invocation.

    Returns:
        ``True`` when the path is likely produced by a prior data-generation job.
    """
    normalized = value.replace("\\", "/")
    return normalized.startswith(
        ("data/processed/train_dpo", "data/processed/train_temporal")
    )


def _is_connectivity_error(exception: Exception) -> bool:
    """Return whether an exception appears to be network/connectivity-related.

    Args:
        exception: Exception raised while validating remote model IDs.

    Returns:
        ``True`` when error text suggests connectivity problems.
    """
    message = str(exception).lower()
    hints = [
        "couldn't connect",
        "connection error",
        "connection aborted",
        "max retries exceeded",
        "name or service not known",
        "temporary failure",
        "timed out",
    ]
    return any(hint in message for hint in hints)


@lru_cache(maxsize=64)
def _validate_remote_model_id(model_id: str) -> None:
    """Validate that a Hugging Face model ID exists.

    Args:
        model_id: Hugging Face model ID.
    """
    AutoConfig.from_pretrained(model_id)


def _value_is_runtime_resolved(value: str) -> bool:
    """Return True when the value contains an unexpanded shell variable.

    `_resolve_repo_path` already strips well-known BSUB-preamble variables
    such as ``$EXPERIMENT_DIR``. Anything else with a ``$`` (e.g. a function-
    local ``$model_path`` or a top-level scalar that we couldn't parse) only
    resolves at runtime — static existence checks against it are not
    meaningful.
    """
    return "$" in value


def _validate_paths(invocation: JobInvocation) -> None:
    """Validate local path options for an invocation.

    Args:
        invocation: Parsed command invocation.
    """
    for option, value in _iter_option_values(invocation.args):
        if option in EXISTING_FILE_OPTIONS:
            resolved = _resolve_repo_path(value)
            if _value_is_runtime_resolved(str(resolved)):
                continue
            if (
                not resolved.is_file()
                and _looks_like_generated_input_artifact(value)
                and not STRICT_GENERATED_INPUTS
            ):
                continue
            assert resolved.is_file(), (
                f"{invocation.job_script}:{invocation.line_number} -> {option} "
                f"points to missing file: {value}"
            )

        if option in EXISTING_DIR_OPTIONS:
            resolved = _resolve_repo_path(value)
            if _value_is_runtime_resolved(str(resolved)):
                continue
            assert resolved.is_dir(), (
                f"{invocation.job_script}:{invocation.line_number} -> {option} "
                f"points to missing directory: {value}"
            )


def _validate_models(invocation: JobInvocation) -> None:
    """Validate model references used by an invocation.

    Args:
        invocation: Parsed command invocation.
    """
    global REMOTE_VALIDATION_SKIP_REASON

    for option, value in _iter_option_values(invocation.args):
        if option not in MODEL_OPTIONS:
            continue

        if _value_is_runtime_resolved(value):
            # Function-local bash variables (e.g. `--model-path "$model_path"`
            # inside a shell function) can't be statically resolved.
            continue

        if _looks_like_local_model_reference(value):
            local_path = _resolve_repo_path(value)
            if STRICT_LOCAL_MODEL_PATHS:
                assert local_path.exists(), (
                    f"{invocation.job_script}:{invocation.line_number} -> {option} "
                    f"expects local path that does not exist: {value}"
                )
            continue

        if not VALIDATE_REMOTE_MODELS:
            continue

        if REMOTE_VALIDATION_SKIP_REASON:
            pytest.skip(REMOTE_VALIDATION_SKIP_REASON)

        try:
            _validate_remote_model_id(value)
        except Exception as exc:  # pragma: no cover - branch depends on network
            if _is_connectivity_error(exc):
                REMOTE_VALIDATION_SKIP_REASON = (
                    "Skipping remote model validation due to connectivity issues."
                )
                pytest.skip(REMOTE_VALIDATION_SKIP_REASON)
            pytest.fail(
                f"{invocation.job_script}:{invocation.line_number} -> invalid model "
                f"reference for {option}: {value}\nError: {exc}"
            )


def _invocation_id(invocation: JobInvocation) -> str:
    """Return readable pytest param ID for an invocation."""
    relative = invocation.job_script.relative_to(PROJECT_ROOT)
    return f"{relative}:{invocation.line_number}:{invocation.target}"


JOB_SCRIPTS = _list_job_scripts()
JOB_INVOCATIONS = _collect_job_invocations()


@pytest.mark.parametrize(
    "job_script",
    JOB_SCRIPTS,
    ids=[str(path.relative_to(PROJECT_ROOT)) for path in JOB_SCRIPTS],
)
def test_job_scripts_have_valid_shell_syntax(job_script: Path) -> None:
    """Ensure all job scripts pass shell syntax checks."""
    result = subprocess.run(
        ["bash", "-n", str(job_script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Shell syntax error in {job_script}:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_job_invocation_discovery_is_non_empty() -> None:
    """Guard against silent parser regressions returning no invocations."""
    assert JOB_INVOCATIONS, "No Python invocations were discovered in job scripts."


@pytest.mark.parametrize("invocation", JOB_INVOCATIONS, ids=_invocation_id)
def test_job_python_targets_exist(invocation: JobInvocation) -> None:
    """Ensure Python targets referenced by jobs exist in the repository."""
    target_path = _resolve_repo_path(invocation.target)
    assert target_path.is_file(), (
        f"{invocation.job_script}:{invocation.line_number} references missing "
        f"Python target: {invocation.target}"
    )


@pytest.mark.parametrize(
    "invocation",
    [
        inv
        for inv in JOB_INVOCATIONS
        if inv.target.startswith(("src/asa/", "scripts/train/", "scripts/eval/",
                                  "scripts/data/", "scripts/diagnostics/"))
    ],
    ids=_invocation_id,
)
def test_asa_job_arguments_parse_with_typer(invocation: JobInvocation) -> None:
    """Validate CLI options for ASA entrypoints using Typer parsing only."""
    # If any arg still holds an unexpanded shell variable that our static
    # parser couldn't substitute (function-local var, runtime-computed value,
    # or a scalar with a heredoc/$()-spanning RHS we skipped), typer's type
    # parsing will reject it (e.g. `int($TOTAL_MIX_FILES)`). The job will be
    # well-formed at submission time; we just can't validate it statically.
    if any("$" in arg for arg in invocation.args):
        pytest.skip(
            "Skipping typer parse: invocation contains unexpanded shell "
            "variables that can't be resolved statically."
        )

    command = _load_typer_command(invocation.target)
    try:
        command.make_context(
            info_name=Path(invocation.target).name,
            args=list(invocation.args),
            resilient_parsing=False,
        )
    except Exception as exc:
        pytest.fail(
            f"Failed to parse args for {invocation.target} from "
            f"{invocation.job_script}:{invocation.line_number}\n"
            f"Raw command: {invocation.raw}\n"
            f"Args: {list(invocation.args)}\n"
            f"Error: {exc}"
        )

    _validate_paths(invocation)
    _validate_models(invocation)
