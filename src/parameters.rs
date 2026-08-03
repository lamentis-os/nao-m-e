/// Fixed-point denominator representing the unit value.
pub const SCALE: u32 = 1_000_000;

/// Fraction of relevance weight projected into a recall score.
pub const PROPAGATION_GAIN_PPM: u32 = 400_000;

/// Maximum cue-derived contribution to a recall score.
pub const STRUCTURAL_GAIN_PPM: u32 = 400_000;

/// Preferred relevance change applied to each target in one feedback event.
pub const FEEDBACK_TARGET_STEP_PPM: u32 = 1_000;

/// Maximum aggregate direct target adjustment in one feedback event.
pub const FEEDBACK_MAX_EVENT_PPM: u32 = 10_000;

/// Maximum number of target entries accepted by one feedback event.
///
/// This equals [`FEEDBACK_MAX_EVENT_PPM`] so every accepted target can receive
/// at least one ppm before the remaining outgoing capacity is considered.
pub const MAX_FEEDBACK_TARGETS: usize = FEEDBACK_MAX_EVENT_PPM as usize;

const _: () = assert!(PROPAGATION_GAIN_PPM <= SCALE);
const _: () = assert!(STRUCTURAL_GAIN_PPM + PROPAGATION_GAIN_PPM <= SCALE);
const _: () = assert!(FEEDBACK_TARGET_STEP_PPM <= FEEDBACK_MAX_EVENT_PPM);
const _: () = assert!(FEEDBACK_MAX_EVENT_PPM <= SCALE);
