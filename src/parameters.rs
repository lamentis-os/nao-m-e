/// Fixed-point unit representing one.
pub const SCALE: u32 = 1_000_000;

/// Activation retained by an atom during one logical step.
pub const RETENTION_PPM: u32 = 500_000;

/// Activation made available for propagation during one logical step.
pub const PROPAGATION_GAIN_PPM: u32 = 400_000;

/// Squared fixed-point scale used as the propagation denominator.
pub(crate) const SCALE_SQUARED: u64 = SCALE as u64 * SCALE as u64;

/// Transition numerator at which an activation necessarily saturates.
pub(crate) const SCALE_CUBED: u64 = SCALE_SQUARED * SCALE as u64;

const _: () = assert!(RETENTION_PPM <= SCALE);
const _: () = assert!(PROPAGATION_GAIN_PPM <= SCALE);
const _: () = assert!(RETENTION_PPM + PROPAGATION_GAIN_PPM < SCALE);
