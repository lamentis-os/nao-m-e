/// Fixed-point denominator representing the unit value.
pub const SCALE: u32 = 1_000_000;

/// Maximum cue-derived contribution to a recall score.
pub const STRUCTURAL_GAIN_PPM: u32 = 400_000;

/// Maximum signed learned contribution to a recall score.
pub const LEARNED_GAIN_PPM: u32 = 400_000;

/// Number of recent binary samples retained by one feedback trace.
pub const FEEDBACK_HISTORY_CAPACITY: u8 = 16;

/// Prior mass that tempers short feedback histories.
pub const FEEDBACK_PRIOR_MASS: u8 = 7;

/// Maximum number of target entries accepted by one feedback event.
pub const MAX_FEEDBACK_TARGETS: usize = 10_000;

const _: () = assert!(FEEDBACK_HISTORY_CAPACITY > 0);
const _: () = assert!(FEEDBACK_HISTORY_CAPACITY <= u16::BITS as u8);
const _: () = assert!(STRUCTURAL_GAIN_PPM + LEARNED_GAIN_PPM <= SCALE);
