// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::{self, Write};

/// Print the program name and crate version, then the kernel adapter this
/// process selected.
pub(crate) fn run(out: &mut impl Write) -> io::Result<()> {
    writeln!(out, "rusticide {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(out, "kernels: {}", rusticide_kernels::active().name())
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn version_names_the_kernels() {
        let mut out = Vec::new();
        run(&mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            format!(
                "rusticide {}\nkernels: {}\n",
                env!("CARGO_PKG_VERSION"),
                rusticide_kernels::active().name()
            )
        );
    }
}
