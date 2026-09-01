use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodecErrorKind {
    InvalidInput,
    UnsupportedFormat,
    LimitExceeded,
    Decode,
    Encode,
}

#[derive(Debug)]
pub struct CodecError {
    kind: CodecErrorKind,
    message: String,
}

impl CodecError {
    pub(crate) fn new(kind: CodecErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> CodecErrorKind {
        self.kind
    }
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CodecError {}
