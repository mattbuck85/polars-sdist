# Contributing to polars-sdist

## Architecture

polars-sdist is a [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs) Polars plugin. Rust code in `src/` implements statistical distributions via [statrs](https://github.com/statrs-dev/statrs), exposed to Python through `pyo3-polars` as native Polars expressions.

```
src/            Rust plugin implementation
python/         Python wrapper (polars_sdist namespace)
tests/          pytest suite
benchmarks/     Performance comparison vs scipy
```

## Prerequisites

- Python >= 3.9
- Rust stable toolchain (`rustup default stable`)
- maturin >= 1.7 (`pip install maturin`)

## Local development

```bash
# Create venv and install in dev mode (compiles Rust on first run)
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest scipy

# Build and install the extension
maturin develop --release

# Run tests
pytest tests/ -v
```

`maturin develop` builds the Rust code and installs the resulting `.so` directly into your venv. Use `--release` for optimized builds (slower compile, faster runtime).

## How the build works

### maturin + PyO3

maturin compiles `src/` into a native Python extension module (`.so` / `.pyd`). Key config:

| File | Purpose |
|-|-|
| `Cargo.toml` | Rust dependencies, crate type (`cdylib`), PyO3 features |
| `pyproject.toml` | Python metadata, maturin settings (`module-name`, `python-source`) |

The `pyo3/abi3-py39` feature in `Cargo.toml` builds against Python's [stable ABI](https://docs.python.org/3/c-api/stable.html). This means a single compiled wheel works on Python 3.9+, rather than needing separate builds per Python version.

### Wheel tags

A wheel built with abi3 gets tagged `cp39-abi3-{platform}`, e.g.:

```
polars_sdist-0.1.0-cp39-abi3-manylinux_2_17_x86_64.whl
polars_sdist-0.1.0-cp39-abi3-macosx_11_0_arm64.whl
```

The `cp39` means "minimum Python 3.9", and `abi3` means any Python >= 3.9 can use it.

## CI

### Tests (`.github/workflows/ci.yml`)

Runs on push to `main` and PRs:

- **test** — matrix of Python 3.9 + 3.14 on ubuntu. Builds from source with `maturin build --release`, installs the wheel, runs pytest.
- **lint** — `cargo fmt --check`, `cargo clippy`, `pyright` on the Python wrapper.

### Release (`.github/workflows/release.yml`)

Triggered by pushing a `v*` tag (e.g. `v0.1.0`):

1. **build** — uses `PyO3/maturin-action` to cross-compile wheels for 5 targets:

   | Runner | Target |
   |-|-|
   | `ubuntu-latest` | `x86_64-unknown-linux-gnu` |
   | `ubuntu-latest` | `aarch64-unknown-linux-gnu` |
   | `macos-latest` | `x86_64-apple-darwin` |
   | `macos-latest` | `aarch64-apple-darwin` |
   | `windows-latest` | `x86_64-pc-windows-msvc` |

   The `manylinux: auto` setting runs the build inside a manylinux container so the resulting wheel is compatible with most Linux distros (glibc >= 2.17). maturin auto-detects `abi3-py39` from `Cargo.toml` so each target produces one universal wheel.

2. **sdist** — builds a source distribution (`.tar.gz`) for users without a prebuilt wheel.

3. **publish** — downloads all artifacts and publishes to PyPI via trusted publishing (`id-token: write`).

### Cutting a release

```bash
# Tag and push
git tag v0.2.0
git push origin v0.2.0
```

The release workflow builds all wheels and publishes to PyPI automatically.

## Rust lint

```bash
cargo fmt --check
cargo clippy -- -D warnings
```

## Python type checking

```bash
pip install pyright polars
pyright python/polars_sdist/
```
