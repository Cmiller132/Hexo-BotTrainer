//! Shared record, sample-store, and state utility code for Hexo.
//!
//! Model-owned Rust crates own search, encoding, and sample generation. This
//! crate keeps only stable utilities that are intentionally shared across
//! training, runner, and model packages.

pub mod records;
pub mod state_hash;

#[cfg(feature = "python")]
pub mod pybridge;

pub use records::{
    AbortRecord, HexoRecord, HexoRecordEngineMetadata, HexoRecordFile, HexoRecordFileMode,
    HexoRecordGameWriter, HexoRecordPlayer, HexoRecordRef, HexoRecordStatus,
    RecordError as HexoRecordError, HEXO_RECORD_MAGIC, HEXO_RECORD_SCHEMA_VERSION,
};
pub use state_hash::{hash_state, StateHash};
