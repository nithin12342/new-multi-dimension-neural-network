//! SafeTensors Checkpoint Remapper & Schema Validator
//! Validates tensor shapes and layer aliases at the binary boundary to prevent silent weight resets.

use safetensors::tensor::SafeTensors;
use std::collections::HashMap;

pub struct StateDictRemapper {
    pub canonical_shapes: HashMap<String, Vec<usize>>,
    pub key_aliases: HashMap<String, String>,
}

impl StateDictRemapper {
    pub fn new(canonical_shapes: HashMap<String, Vec<usize>>, key_aliases: HashMap<String, String>) -> Self {
        Self { canonical_shapes, key_aliases }
    }

    pub fn resolve_key(&self, raw_key: &str) -> String {
        self.key_aliases.get(raw_key).cloned().unwrap_or_else(|| raw_key.to_string())
    }

    pub fn verify_tensor_layout(&self, key: &str, shape: &[usize]) -> Result<(), String> {
        let canonical_key = self.resolve_key(key);
        match self.canonical_shapes.get(&canonical_key) {
            Some(expected) if expected.as_slice() == shape => Ok(()),
            Some(expected) => Err(format!("Shape mismatch on {canonical_key}: expected {expected:?}, found {shape:?}")),
            None => Err(format!("Unrecognized model parameter: {key}")),
        }
    }
}

pub struct CheckpointValidator {
    expected_shapes: HashMap<String, Vec<usize>>,
    alias_map: HashMap<String, String>,
}

impl CheckpointValidator {
    pub fn new(expected: HashMap<String, Vec<usize>>) -> Self {
        let mut alias_map = HashMap::new();
        alias_map.insert("decoder.classifier.weight".to_string(), "decoder.gyroplane.centroids".to_string());
        alias_map.insert("model.decoder.classifier.weight".to_string(), "model.decoder.gyroplane.centroids".to_string());
        alias_map.insert("classifier.weight".to_string(), "gyroplane.centroids".to_string());

        Self {
            expected_shapes: expected,
            alias_map,
        }
    }

    /// Add custom alias mapping
    pub fn add_alias(&mut self, source: &str, target: &str) {
        self.alias_map.insert(source.to_string(), target.to_string());
    }

    /// Validates binary SafeTensors buffer against expected parameter shapes
    pub fn validate_and_remap(&self, buffer: &[u8]) -> Result<HashMap<String, Vec<usize>>, String> {
        let tensors = SafeTensors::deserialize(buffer)
            .map_err(|e| format!("SafeTensors deserialization error: {e}"))?;

        let mut discovered_tensors = HashMap::new();

        for (name, expected_shape) in &self.expected_shapes {
            // Check direct name or mapped alias
            let tensor_view = tensors.tensor(name)
                .or_else(|_| {
                    // Try reverse alias lookup
                    for (alias_src, alias_dst) in &self.alias_map {
                        if alias_dst == name {
                            if let Ok(view) = tensors.tensor(alias_src) {
                                return Ok(view);
                            }
                        }
                    }
                    Err(safetensors::SafeTensorError::InvalidHeader)
                });

            match tensor_view {
                Ok(view) => {
                    if view.shape() != expected_shape.as_slice() {
                        return Err(format!(
                            "Shape mismatch on parameter '{name}': expected {expected_shape:?}, found {:?}",
                            view.shape()
                        ));
                    }
                    discovered_tensors.insert(name.clone(), view.shape().to_vec());
                }
                Err(_) => {
                    return Err(format!("Fatal: Missing checkpoint parameter key: '{name}'"));
                }
            }
        }

        Ok(discovered_tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_validator_initialization() {
        let mut expected = HashMap::new();
        expected.insert("encoder.weights".to_string(), vec![16, 16]);

        let validator = CheckpointValidator::new(expected);
        assert!(validator.alias_map.contains_key("classifier.weight"));
    }
}
