import pathlib
import shutil

import nox

_ROOT_DIR = pathlib.Path(__file__).parent
_DOCS_DIR = _ROOT_DIR / "docs"
_SRC_DIR = _ROOT_DIR / "src" / "eeg_bci_transformer_pipeline"
_TESTS_DIR = _ROOT_DIR / "tests"
_LINT_DIRS = ("src", "tests")

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["ruff", "test"]


@nox.session(venv_backend="none")
def cleanup(session: nox.Session) -> None:
    """Cleanup temporary files."""
    # Remove all the __pycache__ folders.
    for folder_path in (_ROOT_DIR, _SRC_DIR, _TESTS_DIR):
        _path = folder_path / "__pycache__"
        if _path.exists():
            shutil.rmtree(_path)
    # Cleanup the docs.
    _path = _DOCS_DIR / "_build"
    if _path.exists():
        shutil.rmtree(_path)

# Session to create the documentation locally --> nox -s docs
@nox.session
def docs(session: nox.Session) -> None:
    """Generate docs."""
    session.install(".[docs]")
    source_dir = _DOCS_DIR
    output_dir = _DOCS_DIR / "_build" / "html"
    session.run("sphinx-build", "-b", "html", source_dir, output_dir, *session.posargs)

# Session to run the unit test suite --> nox -s test
@nox.session
def ruff(session: nox.Session) -> None:
    """Run ruff."""
    session.install("ruff")
    session.run("ruff", "check", *_LINT_DIRS, *session.posargs)


# Session to run pylint --> nox -s pylint
@nox.session
def pylint(session: nox.Session) -> None:
    """Run pylint."""
    session.install("pylint")
    session.install("-e", ".[dev]")
    session.run("pylint", *_LINT_DIRS, *session.posargs)

# Session to run the unit test suite --> nox -s test
@nox.session
def test(session: nox.Session) -> None:
    """Run the unit tests."""
    session.install(".[dev]")
    session.run("pytest", *session.posargs)
