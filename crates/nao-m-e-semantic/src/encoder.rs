use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use hf_hub::{HFClientSync, HFRepositorySync, RepoTypeModel};
use ort::environment::Environment;
use ort::memory::DeviceType;
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::value::TensorRef;
use sha2::{Digest, Sha256};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::profile::{
    MODEL_FILE, MODEL_NAME, MODEL_OWNER, MODEL_REVISION, MODEL_SHA256, TOKENIZER_FILE,
    TOKENIZER_SHA256,
};
use crate::{
    CueText, EMBEDDING_DIMENSIONS, Embedding, MAX_EMBEDDING_BATCH_SIZE, QueryText, SemanticError,
};

const MAX_TOKEN_COUNT: usize = 512;

#[derive(Clone, Copy)]
struct ArtifactSpec {
    file: &'static str,
    sha256: [u8; 32],
}

const MODEL: ArtifactSpec = ArtifactSpec {
    file: MODEL_FILE,
    sha256: MODEL_SHA256,
};
const TOKENIZER: ArtifactSpec = ArtifactSpec {
    file: TOKENIZER_FILE,
    sha256: TOKENIZER_SHA256,
};

struct EncoderRuntime {
    tokenizer: Tokenizer,
    session: Session,
}

impl EncoderRuntime {
    fn load() -> Result<Self, SemanticError> {
        let repository = pinned_repository(MODEL_FILE)?;
        let model_path = cached_artifact(&repository, MODEL)?;
        let tokenizer_path = cached_artifact(&repository, TOKENIZER)?;

        let tokenizer = load_tokenizer(&tokenizer_path)?;
        let environment = Environment::current()?;
        let mut cpu_device = None;
        for device in environment.devices() {
            if device.ep()? == "CPUExecutionProvider"
                && device.hardware_device().ty() == DeviceType::CPU
                && cpu_device.replace(device).is_some()
            {
                return Err(SemanticError::RuntimeConfiguration {
                    detail: "multiple canonical CPU execution devices are available",
                });
            }
        }
        let cpu_device = cpu_device.ok_or(SemanticError::RuntimeConfiguration {
            detail: "the canonical CPU execution device is unavailable",
        })?;
        let cpu_options = [("CPUExecutionProvider.use_arena".to_owned(), "0".to_owned())];
        let session = Session::builder()?
            .with_no_environment_execution_providers()
            .map_err(session_builder_error)?
            .with_devices([cpu_device], Some(&cpu_options))
            .map_err(session_builder_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(session_builder_error)?
            .with_parallel_execution(false)
            .map_err(session_builder_error)?
            .with_intra_threads(1)
            .map_err(session_builder_error)?
            .with_inter_threads(1)
            .map_err(session_builder_error)?
            .with_deterministic_compute(true)
            .map_err(session_builder_error)?
            .with_memory_pattern(false)
            .map_err(session_builder_error)?
            .commit_from_file(model_path)?;

        Ok(Self { tokenizer, session })
    }

    fn encode_projected(&mut self, projected: String) -> Result<Embedding, SemanticError> {
        let encoding = self
            .tokenizer
            .encode(projected, true)
            .map_err(tokenizer_error)?;
        let sequence_length = encoding.len();
        if sequence_length == 0 {
            return Err(SemanticError::InvalidModelOutput {
                detail: "tokenizer returned an empty sequence",
            });
        }

        let input_ids = encoding
            .get_ids()
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();
        let attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();
        let token_type_ids = encoding
            .get_type_ids()
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();

        let shape = [1, sequence_length];
        let input_ids = TensorRef::from_array_view((shape, input_ids.as_slice()))?;
        let attention = TensorRef::from_array_view((shape, attention_mask.as_slice()))?;
        let token_types = TensorRef::from_array_view((shape, token_type_ids.as_slice()))?;
        let outputs = self.session.run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention,
            "token_type_ids" => token_types,
        })?;
        let output = outputs
            .get("last_hidden_state")
            .ok_or(SemanticError::InvalidModelOutput {
                detail: "last_hidden_state output is absent",
            })?;
        let (output_shape, values) = output.try_extract_tensor::<f32>()?;
        if output_shape.as_ref()
            != [
                1,
                i64::try_from(sequence_length).expect("token bound fits i64"),
                i64::try_from(EMBEDDING_DIMENSIONS).expect("dimension fits i64"),
            ]
        {
            return Err(SemanticError::InvalidModelOutput {
                detail: "last_hidden_state has an unexpected shape",
            });
        }

        pool_embedding(values, &attention_mask, sequence_length)
    }
}

fn pinned_repository(
    error_file: &'static str,
) -> Result<HFRepositorySync<RepoTypeModel>, SemanticError> {
    let client = HFClientSync::new().map_err(|source| SemanticError::ArtifactUnavailable {
        file: error_file,
        source,
    })?;
    Ok(client.model(MODEL_OWNER, MODEL_NAME))
}

fn cached_artifact(
    repository: &HFRepositorySync<RepoTypeModel>,
    artifact: ArtifactSpec,
) -> Result<PathBuf, SemanticError> {
    verified_artifact(artifact, || {
        repository
            .download_file()
            .filename(artifact.file)
            .revision(MODEL_REVISION)
            .local_files_only(true)
            .send()
            .map_err(|source| SemanticError::ArtifactUnavailable {
                file: artifact.file,
                source,
            })
    })
}

/// Lazy local encoder for the fixed E5 Small profile.
///
/// Construction performs no filesystem or network I/O. The pinned model and
/// tokenizer must already be provisioned in the local Hugging Face cache. The
/// first non-empty request verifies and loads them without network fallback.
pub struct SemanticEncoder {
    runtime: Option<EncoderRuntime>,
}

impl SemanticEncoder {
    /// Creates an unloaded encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self { runtime: None }
    }

    /// Returns whether the model and tokenizer are loaded in this process.
    #[must_use]
    pub const fn is_loaded(&self) -> bool {
        self.runtime.is_some()
    }

    /// Encodes one ordered batch of bound key/value cues.
    ///
    /// Empty input returns an empty vector without loading artifacts. Inputs
    /// larger than [`MAX_EMBEDDING_BATCH_SIZE`] fail before any model work.
    pub fn encode(&mut self, cues: &[CueText<'_>]) -> Result<Vec<Embedding>, SemanticError> {
        if cues.len() > MAX_EMBEDDING_BATCH_SIZE {
            return Err(SemanticError::BatchTooLarge {
                maximum: MAX_EMBEDDING_BATCH_SIZE,
                found: cues.len(),
            });
        }
        if cues.is_empty() {
            return Ok(Vec::new());
        }
        let runtime = self.runtime()?;
        cues.iter()
            .map(|cue| runtime.encode_projected(cue.project()))
            .collect()
    }

    /// Encodes one normalized free-text retrieval query.
    ///
    /// The query is projected with the fixed E5 `query:` prefix and uses the
    /// same lazy model, tokenizer, pooling, normalization, and quantization as
    /// bound cue encoding.
    pub fn encode_query(&mut self, query: QueryText<'_>) -> Result<Embedding, SemanticError> {
        self.runtime()?.encode_projected(query.project())
    }

    fn runtime(&mut self) -> Result<&mut EncoderRuntime, SemanticError> {
        if self.runtime.is_none() {
            self.runtime = Some(EncoderRuntime::load()?);
        }
        Ok(self.runtime.as_mut().expect("runtime was initialized"))
    }
}

impl Default for SemanticEncoder {
    fn default() -> Self {
        Self::new()
    }
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, SemanticError> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(tokenizer_error)?;
    let pad_id = tokenizer
        .token_to_id("<pad>")
        .ok_or(SemanticError::InvalidModelOutput {
            detail: "tokenizer has no <pad> token",
        })?;
    if pad_id != 1 {
        return Err(SemanticError::InvalidModelOutput {
            detail: "tokenizer <pad> identifier is not canonical",
        });
    }
    tokenizer
        .with_truncation(Some(truncation_params()))
        .map_err(tokenizer_error)?;
    tokenizer.with_padding(Some(padding_params(pad_id)));
    Ok(tokenizer)
}

fn truncation_params() -> TruncationParams {
    TruncationParams {
        max_length: MAX_TOKEN_COUNT,
        ..TruncationParams::default()
    }
}

fn padding_params(pad_id: u32) -> PaddingParams {
    PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id,
        pad_token: "<pad>".to_owned(),
        ..PaddingParams::default()
    }
}

fn tokenizer_error(error: impl std::fmt::Display) -> SemanticError {
    SemanticError::Tokenizer {
        detail: error.to_string(),
    }
}

fn session_builder_error(error: ort::Error<SessionBuilder>) -> SemanticError {
    SemanticError::Runtime(error.into())
}

fn verified_artifact(
    artifact: ArtifactSpec,
    resolve: impl FnOnce() -> Result<PathBuf, SemanticError>,
) -> Result<PathBuf, SemanticError> {
    let path = resolve()?;
    let found = sha256_file(artifact.file, &path)?;
    if found == artifact.sha256 {
        Ok(path)
    } else {
        Err(SemanticError::ArtifactHashMismatch {
            file: artifact.file,
            expected: artifact.sha256,
            found,
        })
    }
}

fn sha256_file(file: &'static str, path: &Path) -> Result<[u8; 32], SemanticError> {
    let mut source =
        File::open(path).map_err(|source| SemanticError::ArtifactRead { file, source })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = source
            .read(&mut buffer)
            .map_err(|source| SemanticError::ArtifactRead { file, source })?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest.finalize().into())
}

fn pool_embedding(
    hidden: &[f32],
    attention_mask: &[i64],
    sequence_length: usize,
) -> Result<Embedding, SemanticError> {
    let expected_hidden = sequence_length.checked_mul(EMBEDDING_DIMENSIONS).ok_or(
        SemanticError::InvalidModelOutput {
            detail: "model output size overflowed",
        },
    )?;
    if hidden.len() != expected_hidden || attention_mask.len() != sequence_length {
        return Err(SemanticError::InvalidModelOutput {
            detail: "model output length does not match its shape",
        });
    }

    let mut pooled = [0.0_f64; EMBEDDING_DIMENSIONS];
    let mut token_count = 0_u32;
    for (token, &attention) in attention_mask.iter().enumerate() {
        if attention == 0 {
            continue;
        }
        token_count += 1;
        let start = token * EMBEDDING_DIMENSIONS;
        for (sum, value) in pooled
            .iter_mut()
            .zip(&hidden[start..start + EMBEDDING_DIMENSIONS])
        {
            if !value.is_finite() {
                return Err(SemanticError::InvalidModelOutput {
                    detail: "model output contains a non-finite component",
                });
            }
            *sum += f64::from(*value);
        }
    }
    if token_count == 0 {
        return Err(SemanticError::InvalidModelOutput {
            detail: "attention mask contains no input token",
        });
    }
    let divisor = f64::from(token_count);
    for value in &mut pooled {
        *value /= divisor;
    }
    let norm = pooled.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(SemanticError::InvalidModelOutput {
            detail: "pooled model output has no finite norm",
        });
    }
    let values = pooled
        .iter()
        .map(|value| quantize_component(value / norm))
        .collect::<Result<Vec<_>, _>>()?;
    Embedding::new(values).ok_or(SemanticError::InvalidModelOutput {
        detail: "normalized model output quantized to zero",
    })
}

fn quantize_component(normalized: f64) -> Result<i16, SemanticError> {
    let scaled = normalized * 32_767.0;
    if !scaled.is_finite() {
        return Err(SemanticError::InvalidModelOutput {
            detail: "normalized model output contains a non-finite component",
        });
    }
    Ok(scaled.round().clamp(-32_767.0, 32_767.0) as i16)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fs;

    use sha2::Digest;
    use tempfile::tempdir;

    use super::{
        ArtifactSpec, MODEL, MODEL_REVISION, SemanticEncoder, TOKENIZER, cached_artifact,
        load_tokenizer, padding_params, pinned_repository, pool_embedding, quantize_component,
        sha256_file, truncation_params, verified_artifact,
    };
    use crate::{CueText, EMBEDDING_DIMENSIONS, MAX_EMBEDDING_BATCH_SIZE, SemanticError};

    #[test]
    fn empty_and_oversized_batches_do_not_initialize_the_runtime() {
        let mut encoder = SemanticEncoder::new();
        assert!(encoder.encode(&[]).unwrap().is_empty());
        assert!(!encoder.is_loaded());

        let cues = vec![CueText::new("key", "value"); MAX_EMBEDDING_BATCH_SIZE + 1];
        assert!(matches!(
            encoder.encode(&cues),
            Err(SemanticError::BatchTooLarge {
                maximum: MAX_EMBEDDING_BATCH_SIZE,
                found
            }) if found == MAX_EMBEDDING_BATCH_SIZE + 1
        ));
        assert!(!encoder.is_loaded());
    }

    #[test]
    fn artifact_hash_mismatch_fails_without_repair_or_retry() {
        let directory = tempdir().unwrap();
        let bad = directory.path().join("bad");
        fs::write(&bad, b"bad").unwrap();
        let expected: [u8; 32] = sha2::Sha256::digest(b"good").into();
        let bad_digest: [u8; 32] = sha2::Sha256::digest(b"bad").into();
        let calls = Cell::new(0);

        let result = verified_artifact(
            ArtifactSpec {
                file: "test",
                sha256: expected,
            },
            || {
                calls.set(calls.get() + 1);
                Ok(bad)
            },
        );

        assert!(matches!(
            result,
            Err(SemanticError::ArtifactHashMismatch { found, .. })
                if found == bad_digest
        ));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    #[ignore = "explicitly provisions the pinned model assets for runtime qualification"]
    fn provision_pinned_artifacts_for_runtime_qualification() {
        let repository = pinned_repository(MODEL.file).unwrap();
        for artifact in [MODEL, TOKENIZER] {
            let mut force_download = false;
            loop {
                let path = repository
                    .download_file()
                    .filename(artifact.file)
                    .revision(MODEL_REVISION)
                    .force_download(force_download)
                    .send()
                    .unwrap();
                if sha256_file(artifact.file, &path).unwrap() == artifact.sha256 {
                    break;
                }
                assert!(!force_download, "provisioned artifact hash remains invalid");
                force_download = true;
            }
        }
    }

    #[test]
    fn tokenizer_configuration_is_exact_and_bounded() {
        let truncation = truncation_params();
        assert_eq!(truncation.max_length, 512);
        assert_eq!(truncation.stride, 0);
        assert!(matches!(
            truncation.direction,
            tokenizers::TruncationDirection::Right
        ));
        assert!(matches!(
            truncation.strategy,
            tokenizers::TruncationStrategy::LongestFirst
        ));

        let padding = padding_params(1);
        assert!(matches!(
            padding.strategy,
            tokenizers::PaddingStrategy::BatchLongest
        ));
        assert!(matches!(
            padding.direction,
            tokenizers::PaddingDirection::Right
        ));
        assert_eq!(padding.pad_id, 1);
        assert_eq!(padding.pad_type_id, 0);
        assert_eq!(padding.pad_token, "<pad>");
        assert_eq!(padding.pad_to_multiple_of, None);
    }

    #[test]
    #[ignore = "requires the provisioned pinned 17 MB tokenizer asset"]
    fn pinned_tokenizer_matches_golden_ids_and_right_truncation() {
        let repository = pinned_repository(TOKENIZER.file).unwrap();
        let path = cached_artifact(&repository, TOKENIZER).unwrap();
        let tokenizer = load_tokenizer(&path).unwrap();
        let goldens = [(
            CueText::new("problem", "login returns http 404"),
            &[0, 46692, 12, 2967, 12, 73655, 30646, 7, 1621, 1112, 617, 2][..],
        )];
        for (cue, expected) in goldens {
            let encoding = tokenizer.encode(cue.project(), true).unwrap();
            assert_eq!(encoding.get_ids(), expected);
        }

        let long_value = format!(
            "start {} end",
            std::iter::repeat_n("boundary", 700)
                .collect::<Vec<_>>()
                .join(" ")
        );
        let projected = CueText::new("truncation", &long_value).project();
        let truncated = tokenizer.encode(projected.clone(), true).unwrap();
        assert_eq!(truncated.len(), 512);

        let raw_tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
        let untruncated = raw_tokenizer.encode(projected, true).unwrap();
        assert!(untruncated.len() > truncated.len());
        assert_eq!(&truncated.get_ids()[..511], &untruncated.get_ids()[..511]);
        assert_eq!(truncated.get_ids()[511], 2);
        assert_ne!(untruncated.get_ids()[511], 2);
        assert_eq!(untruncated.get_ids().last(), Some(&2));
    }

    #[test]
    fn pooling_ignores_padding_normalizes_and_rounds_to_i16() {
        let mut hidden = vec![0.0_f32; 2 * EMBEDDING_DIMENSIONS];
        hidden[0] = 3.0;
        hidden[1] = 4.0;
        hidden[EMBEDDING_DIMENSIONS] = 99.0;

        let embedding = pool_embedding(&hidden, &[1, 0], 2).unwrap();
        assert_eq!(embedding.values()[0], 19_660);
        assert_eq!(embedding.values()[1], 26_214);
        assert!(embedding.values()[2..].iter().all(|value| *value == 0));
    }

    #[test]
    fn pooling_rejects_non_finite_and_zero_vectors() {
        let zero = vec![0.0_f32; EMBEDDING_DIMENSIONS];
        assert!(matches!(
            pool_embedding(&zero, &[1], 1),
            Err(SemanticError::InvalidModelOutput { .. })
        ));
        let mut invalid = zero;
        invalid[7] = f32::NAN;
        assert!(matches!(
            pool_embedding(&invalid, &[1], 1),
            Err(SemanticError::InvalidModelOutput { .. })
        ));
        let infinity = vec![f32::INFINITY; EMBEDDING_DIMENSIONS];
        assert!(matches!(
            pool_embedding(&infinity, &[1], 1),
            Err(SemanticError::InvalidModelOutput { .. })
        ));
    }

    #[test]
    fn quantization_rounds_half_ties_away_and_clamps_symmetrically() {
        let half_step = 0.5 / 32_767.0;
        assert_eq!(quantize_component(half_step).unwrap(), 1);
        assert_eq!(quantize_component(-half_step).unwrap(), -1);
        assert_eq!(quantize_component(1.5 / 32_767.0).unwrap(), 2);
        assert_eq!(quantize_component(-1.5 / 32_767.0).unwrap(), -2);
        assert_eq!(quantize_component(2.0).unwrap(), 32_767);
        assert_eq!(quantize_component(-2.0).unwrap(), -32_767);
        assert!(matches!(
            quantize_component(f64::INFINITY),
            Err(SemanticError::InvalidModelOutput { .. })
        ));
        assert!(matches!(
            quantize_component(f64::NAN),
            Err(SemanticError::InvalidModelOutput { .. })
        ));
    }
}
