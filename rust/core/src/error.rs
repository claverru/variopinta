use std::fmt;

#[derive(Debug)]
pub enum CoreError {
    Invalid(String),
    Runtime(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) | Self::Runtime(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for CoreError {}

pub type CoreResult<T> = Result<T, CoreError>;
