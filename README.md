# dora-openarm-evaluation-rust

A Rust port of [`dora-openarm-evaluation`](https://github.com/enactic/dora-openarm-evaluation),
the [dora-rs](https://dora-rs.ai/) evaluation orchestration repository for
[OpenArm](https://openarm.dev/).

Upstream is not a Python package -- it has no `pyproject.toml` and ships no
console scripts. It is dataflow YAML plus a `Dockerfile` and two loose
scripts, `src/local_policy_server.py` and `src/docker_policy_server.py`,
that load a [LeRobot](https://github.com/huggingface/lerobot) ACT policy
checkpoint with PyTorch and sit behind the `AF_UNIX` socket that the
already-published
[`dora-openarm-local-policy-server-rust`](https://github.com/k1000dai/dora-openarm-local-policy-server-rust)
and
[`dora-openarm-docker-policy-server-rust`](https://github.com/k1000dai/dora-openarm-docker-policy-server-rust)
dora nodes talk to.

**This repository does not implement LeRobot or PyTorch in Rust.** No such
runtime exists for either, and reimplementing ACT's architecture, trained
weights, and numerics from scratch is out of scope for a protocol/transport
port. Read ["The LeRobot boundary"](#the-lerobot-boundary) before assuming
otherwise.

What this repository *does* port, byte-for-byte where upstream's own
behavior is deterministic:

- The two Unix-socket **roles** (`local_policy_server.py` binds/listens/
  accepts; `docker_policy_server.py` connects) -- see
  [`src/socket_role.rs`](src/socket_role.rs).
- The **NDJSON request/response protocol** -- see
  [`src/protocol.rs`](src/protocol.rs).
- **Arrow IPC FILE observation loading** -- see
  [`src/observation.rs`](src/observation.rs).
- **Resolution detection** (`detect_resolution`) -- see
  [`src/resolution.rs`](src/resolution.rs).
- **Image resize and channel normalization** (`prepare_image`) -- see
  [`src/image_prep.rs`](src/image_prep.rs).
- The **camera/state key mapping** (`CAMERA_KEY_MAP`) -- see
  [`src/camera_map.rs`](src/camera_map.rs).
- **Observation batch assembly** (`observation_to_batch`) -- see
  [`src/batch.rs`](src/batch.rs).
- The **request/response loop** shared by both servers -- see
  [`src/server.rs`](src/server.rs).
- The **`interval=33_333_333` / `cutoff_hz=15` / `positions` shape**
  contract -- see [`src/protocol.rs`](src/protocol.rs).

Two binaries implement upstream's two scripts, with model inference
plugged in behind a trait:

- `dora-openarm-evaluation-local-policy-server` -- port of
  `src/local_policy_server.py`.
- `dora-openarm-evaluation-docker-policy-server` -- port of
  `src/docker_policy_server.py`.

Ported dataflow YAML (`build:`/`path:` remapped to `-rust` node crates
where one exists) lives under [`dataflows/`](dataflows/), not in the Cargo
package, since dora dataflows are not Rust source. See
[`metadata.yaml`](metadata.yaml) (copied verbatim) and
[`docker/`](docker/) for the two Dockerfiles.

## The LeRobot boundary

Upstream's `infer()` function does exactly one Rust-unfriendly thing:
`policy.predict_action_chunk(batch)`, a forward pass through a trained ACT
checkpoint (`enactic/act-openarm-2-cell-pick_up_cube_mujoco`) via
`lerobot`'s Python API. Everything *around* that call -- socket setup,
protocol framing, observation loading, resolution detection, image
resize/normalization, and response construction -- is ordinary,
portable logic, and this crate ports it.

The seam between the two is [`PolicyModel`](src/policy.rs):

```rust
pub trait PolicyModel {
    fn image_sizes(&self) -> HashMap<String, (u32, u32)>;
    fn infer(&self, batch: &ModelBatch) -> Vec<Vec<f32>>;
}
```

`image_sizes` mirrors upstream's `policy.config.input_features` lookup
(what `(height, width)` each camera input should be resized to); `infer`
mirrors `predict_action_chunk` (`ModelBatch` in, one action-chunk row per
step out).

The [`MockPolicy`](src/policy.rs) shipped here implements this trait
deterministically -- it echoes the observation's state vector, truncated
or zero-padded, into every row of a fixed-shape chunk -- and is what both
binaries use by default and what this crate's own tests assert against.
**It is not a LeRobot ACT implementation.** Its output has no relationship
to trained policy weights. Do not point a real robot at this crate's
binaries expecting meaningful actions.

If you need real inference, you have two options, and this port does not
pick one for you:

1. **Run upstream's own Python sidecar.** [`docker/lerobot-src/`](docker/lerobot-src/)
   holds unmodified copies of upstream's `local_policy_server.py` and
   `docker_policy_server.py`; [`docker/Dockerfile.lerobot`](docker/Dockerfile.lerobot)
   is upstream's own `Dockerfile`, reproduced verbatim (only its `COPY`
   source path is adjusted to match this repository's layout). Building
   and running it needs no changes to the Rust side of this dataflow at
   all -- the dora nodes on the other end of the socket
   (`dora-openarm-local-policy-server-rust` /
   `dora-openarm-docker-policy-server-rust`) don't know or care what
   language answers them.
2. **Implement `PolicyModel` yourself.** A real adapter could shell out to
   a Python subprocess, call an HTTP inference server (the way
   `dora-openarm-classifier`'s `--server-url` mode already does for a
   different model), or bind an ONNX/candle runtime if ACT ever gets a
   supported export. None of those adapters exists in this crate;
   `MockPolicy` is the only implementation provided. Both Rust executables
   refuse to start unless `--mock` is passed explicitly, so this stand-in cannot
   silently emit meaningless actions in a real evaluation run.

## Socket roles

| Process | Role | Counterpart dora node |
|---|---|---|
| `dora-openarm-evaluation-local-policy-server` (this repo) | binds, listens, accepts | `dora-openarm-local-policy-server` **connects** |
| `dora-openarm-evaluation-docker-policy-server` (this repo) | connects | `dora-openarm-docker-policy-server` **binds, listens, accepts** |

Getting either backwards deadlocks the two processes waiting for each
other. Only the local server owns (and removes) its socket file; the
docker server never removes the socket file it connects to, since the
*node* on that side owns it -- both details match upstream exactly. See
[`src/socket_role.rs`](src/socket_role.rs).

## Protocol

One JSON object per line, in both directions, on a single connection:

- **Request** (node → this server):
  `{"name":"inference","data_path":"...","reset":bool,"metadata":{...}}`.
  `name` and `metadata` are accepted but not interpreted, matching
  upstream. `reset` is parsed but deliberately unused: upstream's
  `policy.reset()` runs once at startup, not per request.
- **Response** (this server → node):
  `{"interval":33333333,"cutoff_hz":15,"positions":[[...],...]}`, field
  order matching upstream's dict exactly. `positions` is one row per
  action step, `interval` is nanoseconds between rows (30 Hz), `cutoff_hz`
  is the low-pass filter cutoff in Hz the actions executor applies
  downstream.

See [`src/protocol.rs`](src/protocol.rs) for the implementation and
`tests/protocol.rs` for byte-exact fixtures.

## Preprocessing pipeline

For each request: `load_observation` opens the Arrow IPC **FILE** (not
the streaming format) at `data_path` and reads its single-row
`StructArray`. `build_batch` then extracts the `position` field
verbatim as the state vector, and for each of the three mapped camera
fields (`camera_head_left`, `camera_wrist_left`, `camera_wrist_right` --
`camera_ceiling` and one of the two head/wrist pairings are **not**
consumed, matching upstream's `CAMERA_KEY_MAP` exactly) calls
`prepare_image`: `detect_resolution` recovers `(height, width)` from the
buffer length, the image is resized to the policy's declared input size
if it differs, and converted from channel-last `u8` `[0, 255]` to
channel-first (`CHW`) `f32` `[0.0, 1.0]`.

**Resize fidelity boundary.** Upstream resizes with
`PIL.Image.fromarray(img).resize(...)` (Pillow's default bicubic filter).
This port resizes with the `image` crate's Catmull-Rom filter. The two
are **not** guaranteed to produce byte-identical pixels -- see
[`src/image_prep.rs`](src/image_prep.rs) module docs. Only the identity
case (no resize needed) is byte-exact, and is what this crate's golden
tests assert numerically.

## Dataflow porting

[`dataflows/`](dataflows/) ports upstream's four dataflow YAMLs
(`dataflow-cell-sample.yaml`, `dataflow-local-inference.yaml`,
`dataflow-docker-inference.yaml`, and `dataflow-docker-evaluation.yaml`).
Node ids, inputs, and outputs are
byte-identical to upstream -- dora wiring is language-agnostic. Only
`build:`/`path:` are remapped, and only for node kinds that have a
published `-rust` port:

| Node kind | `-rust` crate |
|---|---|
| `dora-openarm-quitter` | `dora-openarm-quitter-rust` |
| `dora-openarm-observer` | `dora-openarm-observer-rust` |
| `dora-openarm-evaluation-ui` | `dora-openarm-evaluation-ui-rust` |
| `dora-openarm-local-policy-server` | `dora-openarm-local-policy-server-rust` |
| `dora-openarm-docker-policy-server` | `dora-openarm-docker-policy-server-rust` |
| `dora-openarm-actions-executor` | `dora-openarm-actions-executor-rust` |
| `dora-openarm-inference-controller` | `dora-openarm-inference-controller-rust` |
| `dora-openarm-classifier` | `dora-openarm-classifier-rust` |
| `dora-openarm-dataset-recorder` | `dora-openarm-dataset-recorder-rust` |

Left as upstream Python, deliberately: `dora-openarm` (the arm driver,
blocked on the closed-source `openarm-driver` native library),
`dora-openarm-mujoco` (MuJoCo's Python viewer/renderer has no Rust
analogue), and the third-party `opencv-video-capture` /
`dora-opencv-image-splitter` nodes. Each ported YAML file's header comment
says which nodes it leaves as Python and why. `build:` commands assume
this repository is checked out as a sibling of the node crates it
references (`cargo build --release --manifest-path ../dora-openarm-*-rust/Cargo.toml`);
adjust the paths if your checkout layout differs.

## Docker images

- [`docker/Dockerfile.lerobot`](docker/Dockerfile.lerobot) -- upstream's
  real image, verbatim (Python + PyTorch + LeRobot ACT). Build this for
  real inference; see ["The LeRobot boundary"](#the-lerobot-boundary).
- [`docker/Dockerfile.mock`](docker/Dockerfile.mock) -- builds this
  crate's `dora-openarm-evaluation-docker-policy-server` binary into a
  minimal image with **no GPU, no PyTorch, and no baked-in model
  weights**. Useful for exercising the docker socket role and protocol
  end-to-end without either. Build from this repository's root:
  `docker build -f docker/Dockerfile.mock -t openarm-eval-mock:latest .`

## Usage

Build both binaries:

```console
$ cargo build --release
```

Run the local server (binds `/dev/shm/policy-server.socket` by default,
or the given argument):

```console
$ ./target/release/dora-openarm-evaluation-local-policy-server --mock [socket_path]
```

Run the docker server (socket path argument required, matching upstream):

```console
$ ./target/release/dora-openarm-evaluation-docker-policy-server --mock <socket_path>
```

Then run a dataflow from [`dataflows/`](dataflows/):

```console
$ dora build dataflows/dataflow-local-inference.yaml --uv
$ dora run dataflows/dataflow-local-inference.yaml --uv
```

## Migration mapping

| Python (upstream) | Rust (this port) |
|---|---|
| `src/local_policy_server.py` | `src/bin/local_policy_server.rs` (dora-free protocol logic delegated to the library modules below) |
| `src/docker_policy_server.py` | `src/bin/docker_policy_server.rs` |
| `DEFAULT_SOCKET`, `sys.argv[1]` | `src/cli.rs` |
| `detect_resolution` | `src/resolution.rs::detect_resolution` |
| `prepare_image` | `src/image_prep.rs::prepare_image` |
| `CAMERA_KEY_MAP` | `src/camera_map.rs::CAMERA_KEY_MAP` |
| `observation_to_batch` | `src/batch.rs::build_batch` |
| `pa.OSFile` + `pa.ipc.open_file` | `src/observation.rs::load_observation` |
| request/response dicts | `src/protocol.rs::{InferenceRequest, ActionsResponse}` |
| `for line in io: ...` | `src/server.rs::serve_connection` |
| `sock.bind()` / `sock.listen()` / `sock.accept()` | `src/socket_role.rs::bind_and_accept` |
| `sock.connect()` | `src/socket_role.rs::connect` |
| `finally: os.remove(socket_path)` | `src/socket_role.rs::SocketCleanup` |
| `policy.predict_action_chunk(batch)` | `src/policy.rs::PolicyModel::infer` (trait boundary; `MockPolicy` is the only implementation, not a LeRobot port) |
| `pytest` (none exist upstream) | `cargo test --all-targets` |
| `ruff` (none configured upstream) | `cargo fmt` + `cargo clippy` (`pedantic`, `missing_docs`) |

## Development

```console
$ cargo fmt --check
$ cargo clippy --all-targets --all-features -- -D warnings
$ cargo test --all-targets --all-features
$ cargo build --release --all-features
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details,
and [NOTICE](NOTICE) for attribution and scope notes.

Copyright 2026 Enactic, Inc.
