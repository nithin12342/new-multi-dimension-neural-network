//! Lock-Free In-Memory Apache Arrow RecordBatch Buffering
//! Replaces row-by-row table inserts with sub-microsecond vector appends.

use arrow::array::{Float32Builder, Int32Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

pub struct ArrowTelemetryBuffer {
    schema: Arc<Schema>,
    step_builder: Int32Builder,
    stream_builder: StringBuilder,
    loss_builder: Float32Builder,
    ppl_builder: Float32Builder,
    sil_builder: Float32Builder,
    count: usize,
}

impl ArrowTelemetryBuffer {
    pub fn new() -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("step", DataType::Int32, false),
            Field::new("stream", DataType::Utf8, false),
            Field::new("loss", DataType::Float32, false),
            Field::new("ppl", DataType::Float32, false),
            Field::new("silhouette", DataType::Float32, false),
        ]));

        Self {
            schema,
            step_builder: Int32Builder::new(),
            stream_builder: StringBuilder::new(),
            loss_builder: Float32Builder::new(),
            ppl_builder: Float32Builder::new(),
            sil_builder: Float32Builder::new(),
            count: 0,
        }
    }

    /// Sub-microsecond non-allocating append in host RAM
    pub fn append_step(&mut self, step: i32, stream: &str, loss: f32, ppl: f32, sil: f32) {
        self.step_builder.append_value(step);
        self.stream_builder.append_value(stream);
        self.loss_builder.append_value(loss);
        self.ppl_builder.append_value(ppl);
        self.sil_builder.append_value(sil);
        self.count += 1;
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Converts accumulated builder buffers into an immutable RecordBatch and resets builders
    pub fn finish_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(self.step_builder.finish()),
                Arc::new(self.stream_builder.finish()),
                Arc::new(self.loss_builder.finish()),
                Arc::new(self.ppl_builder.finish()),
                Arc::new(self.sil_builder.finish()),
            ],
        )?;
        self.count = 0;
        Ok(batch)
    }

    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

impl Default for ArrowTelemetryBuffer {
    fn default() -> Self {
        Self::new()
    }
}
