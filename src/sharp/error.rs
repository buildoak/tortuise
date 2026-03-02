use std::error::Error;
use std::fmt;

/// Errors that can occur during SHARP model preparation and inference.
#[derive(Debug)]
pub enum SharpError {
    /// Model download or cache population failed.
    Download(String),
    /// File or filesystem I/O failed.
    Io(std::io::Error),
    /// Image decoding or preprocessing failed.
    Image(String),
    /// ONNX Runtime model/session/inference failed.
    Model(String),
    /// SHARP outputs could not be converted into splats.
    PostProcess(String),
}

impl fmt::Display for SharpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SharpError::Download(msg) => write!(f, "download error: {}", msg),
            SharpError::Io(err) => write!(f, "I/O error: {}", err),
            SharpError::Image(msg) => write!(f, "image error: {}", msg),
            SharpError::Model(msg) => write!(f, "model error: {}", msg),
            SharpError::PostProcess(msg) => write!(f, "postprocess error: {}", msg),
        }
    }
}

impl Error for SharpError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            SharpError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SharpError {
    fn from(value: std::io::Error) -> Self {
        SharpError::Io(value)
    }
}
