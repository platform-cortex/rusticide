// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Tracks how much decoded text has been emitted, holding back a trailing
/// U+FFFD that a later token may complete.
#[derive(Debug, Default)]
pub(crate) struct TextStream {
    emitted: usize,
}

impl TextStream {
    /// Returns the part of `decoded` not yet emitted, minus a trailing U+FFFD.
    pub(crate) fn advance<'a>(&mut self, decoded: &'a str) -> &'a str {
        let stable = decoded.strip_suffix('\u{fffd}').unwrap_or(decoded);
        self.take(stable)
    }

    /// Returns everything not yet emitted, including a held-back U+FFFD.
    pub(crate) fn finish<'a>(&mut self, decoded: &'a str) -> &'a str {
        self.take(decoded)
    }

    fn take<'a>(&mut self, text: &'a str) -> &'a str {
        let fresh = text.get(self.emitted..).unwrap_or_default();
        self.emitted += fresh.len();
        fresh
    }
}

#[cfg(test)]
mod tests {
    use super::TextStream;

    fn collect(steps: &[&str], last: &str) -> Vec<String> {
        let mut stream = TextStream::default();
        let mut pieces: Vec<String> = steps
            .iter()
            .map(|decoded| stream.advance(decoded).to_owned())
            .filter(|piece| !piece.is_empty())
            .collect();
        let rest = stream.finish(last);
        if !rest.is_empty() {
            pieces.push(rest.to_owned());
        }
        pieces
    }

    #[test]
    fn streaming_emits_only_the_new_suffix() {
        assert_eq!(collect(&["a", "ab", "abc"], "abc"), ["a", "b", "c"]);
    }

    #[test]
    fn streaming_holds_back_an_incomplete_character() {
        let pieces = collect(&["\u{fffd}", "\u{e1}"], "\u{e1}");
        assert_eq!(pieces, ["\u{e1}"]);
    }

    #[test]
    fn streaming_flushes_a_never_completed_character() {
        let pieces = collect(&["a", "a\u{fffd}"], "a\u{fffd}");
        assert_eq!(pieces, ["a", "\u{fffd}"]);
    }

    #[test]
    fn finish_after_everything_is_emitted_is_empty() {
        assert_eq!(collect(&["a"], "a"), ["a"]);
    }
}
