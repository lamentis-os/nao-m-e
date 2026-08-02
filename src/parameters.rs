/// Fixed-point denominator representing the unit value.
pub const SCALE: u32 = 1_000_000;

/// Fraction of activation retained during one logical step.
pub const RETENTION_PPM: u32 = 500_000;

/// Fraction of weighted activation propagated during one logical step.
pub const PROPAGATION_GAIN_PPM: u32 = 400_000;

/// Preferred relevance change applied to each target in one feedback event.
pub const FEEDBACK_TARGET_STEP_PPM: u32 = 1_000;

/// Maximum total relevance change applied by one feedback event.
pub const FEEDBACK_MAX_EVENT_PPM: u32 = 10_000;

/// Maximum number of target entries accepted by one feedback event.
///
/// This equals [`FEEDBACK_MAX_EVENT_PPM`] so every accepted target can receive
/// at least one ppm before the remaining outgoing capacity is considered.
pub const MAX_FEEDBACK_TARGETS: usize = FEEDBACK_MAX_EVENT_PPM as usize;

/// Squared fixed-point scale used as the propagation denominator.
pub(crate) const SCALE_SQUARED: u64 = SCALE as u64 * SCALE as u64;

/// Transition numerator at which an activation necessarily saturates.
pub(crate) const SCALE_CUBED: u64 = SCALE_SQUARED * SCALE as u64;

const _: () = assert!(RETENTION_PPM <= SCALE);
const _: () = assert!(PROPAGATION_GAIN_PPM <= SCALE);
const _: () = assert!(FEEDBACK_TARGET_STEP_PPM <= FEEDBACK_MAX_EVENT_PPM);
const _: () = assert!(FEEDBACK_MAX_EVENT_PPM <= SCALE);
const _: () = assert!(RETENTION_PPM + PROPAGATION_GAIN_PPM < SCALE);
