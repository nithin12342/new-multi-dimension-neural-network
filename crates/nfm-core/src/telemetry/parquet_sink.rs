//! Epoch-Boundary Snappy Parquet Telemetry Writer
//! Flushes Arrow RecordBatches directly to compressed Parquet files for zero-lock DuckDB querying.

use arrow::array::{Float32Builder, Int32Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use super::ring_buffer::ArrowTelemetryBuffer;

pub struct ArrowTelemetrySink {
    schema: Arc<Schema>,
    step_builder: Int32Builder,
    stream_builder: StringBuilder,
    loss_builder: Float32Builder,
    ppl_builder: Float32Builder,
}

impl ArrowTelemetrySink {
    pub fn new() -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("step", DataType::Int32, false),
            Field::new("stream", DataType::Utf8, false),
            Field::new("loss", DataType::Float32, false),
            Field::new("ppl", DataType::Float32, false),
        ]));

        Self {
            schema,
            step_builder: Int32Builder::new(),
            stream_builder: StringBuilder::new(),
            loss_builder: Float32Builder::new(),
            ppl_builder: Float32Builder::new(),
        }
    }

    #[inline(always)]
    pub fn record_step(&mut self, step: i32, stream: &str, loss: f32, ppl: f32) {
        self.step_builder.append_value(step);
        self.stream_builder.append_value(stream);
        self.loss_builder.append_value(loss);
        self.ppl_builder.append_value(ppl);
    }

    pub fn commit_epoch_parquet(&mut self, destination_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(self.step_builder.finish()),
                Arc::new(self.stream_builder.finish()),
                Arc::new(self.loss_builder.finish()),
                Arc::new(self.ppl_builder.finish()),
            ],
        )?;

        let file = File::create(destination_path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, self.schema.clone(), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }
}

pub struct ParquetTelemetrySink;

impl ParquetTelemetrySink {
    /// Writes an Arrow RecordBatch to a Snappy-compressed Parquet file
    pub fn write_batch<P: AsRef<Path>>(path: P, batch: &RecordBatch) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        writer.write(batch)?;
        writer.close()?;
        Ok(())
    }

    /// Flushes an ArrowTelemetryBuffer directly into a target Parquet file
    pub fn flush_buffer<P: AsRef<Path>>(path: P, buffer: &mut ArrowTelemetryBuffer) -> Result<(), Box<dyn std::error::Error>> {
        let batch = buffer.finish_batch()?;
        Self::write_batch(path, &batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_flush_buffer_to_parquet() {
        let mut buffer = ArrowTelemetryBuffer::new();
        buffer.append_step(1, "stream_1", 1.25, 24.1, 0.65);
        buffer.append_step(2, "stream_1", 1.10, 20.3, 0.71);

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        ParquetTelemetrySink::flush_buffer(&path, &mut buffer).unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }
}
