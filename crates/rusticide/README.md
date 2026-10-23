<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# rusticide

Pure-Rust deep learning framework: the facade crate and its command line.

## Library

```toml
[dependencies]
rusticide = "0.2"
```

```rust
use rusticide::{DType, Device, Tensor};

let t = Tensor::zeros(&[2, 3], DType::F32, Device::cpu())?;
assert_eq!(t.shape().dims(), &[2, 3]);
# Ok::<(), rusticide::Error>(())
```

Cargo features:

- `dispatch` (default): re-exports `rusticide-dispatch` as `rusticide::dispatch`.
- `schema` (default): re-exports `rusticide-schema` as `rusticide::schema`.
- `io`: re-exports `rusticide-io` as `rusticide::io`.
- `cli`: builds the `rusticide` binary (pulls in `clap` and `anyhow`).

`EagerOps` is not re-exported; import it from `rusticide_core`.

## Command line

```sh
cargo install rusticide --features cli
cargo run -p rusticide --features cli -- --help
```

- `rusticide --version` prints `rusticide <version>`.
- `rusticide schema '<operator schema>'` parses a schema and prints its name,
  mutability, arguments and the round-tripped text.
- Running with no arguments prints the usage to stderr and exits with code 2.
