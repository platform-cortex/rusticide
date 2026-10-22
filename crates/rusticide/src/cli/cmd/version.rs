// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::{self, Write};

/// Print the program name and crate version on one line.
pub(crate) fn run(out: &mut impl Write) -> io::Result<()> {
    writeln!(out, "rusticide {}", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn prints_the_name_and_version_line() {
        let mut out = Vec::new();
        run(&mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            format!("rusticide {}\n", env!("CARGO_PKG_VERSION"))
        );
    }
}
