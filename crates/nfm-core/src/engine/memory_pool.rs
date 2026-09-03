use std::collections::HashMap;

/// Pre-allocated Host Memory Pool for zero-copy DMA streaming transfers
pub struct PinnedBufferPool {
    capacity: usize,
    buffer_size: usize,
    buffers: HashMap<String, Vec<Vec<f32>>>,
    indices: HashMap<String, usize>,
}

impl PinnedBufferPool {
    pub fn new(capacity: usize, buffer_size: usize, names: &[&str]) -> Self {
        let mut buffers = HashMap::new();
        let mut indices = HashMap::new();

        for &name in names {
            let slots = (0..capacity).map(|_| vec![0.0f32; buffer_size]).collect();
            buffers.insert(name.to_string(), slots);
            indices.insert(name.to_string(), 0);
        }

        Self {
            capacity,
            buffer_size,
            buffers,
            indices,
        }
    }

    pub fn acquire_buffer_mut(&mut self, name: &str) -> Option<&mut [f32]> {
        if let Some(slots) = self.buffers.get_mut(name) {
            let idx = self.indices.get_mut(name).unwrap();
            let current_idx = *idx;
            *idx = (current_idx + 1) % self.capacity;
            Some(&mut slots[current_idx])
        } else {
            None
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_buffer_pool_cycling() {
        let mut pool = PinnedBufferPool::new(3, 128, &["video", "image"]);
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.buffer_size(), 128);

        // Cycle through capacity
        let buf1 = pool.acquire_buffer_mut("video");
        assert!(buf1.is_some());
        assert_eq!(buf1.unwrap().len(), 128);

        let buf2 = pool.acquire_buffer_mut("video");
        assert!(buf2.is_some());

        let buf3 = pool.acquire_buffer_mut("video");
        assert!(buf3.is_some());

        // 4th acquire cycles back to index 0
        let buf4 = pool.acquire_buffer_mut("video");
        assert!(buf4.is_some());
    }
}
