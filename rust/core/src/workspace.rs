use fast_image_resize as fir;

use crate::kernels::remap::AxisRemapScratch;
use crate::{CoreError, CoreResult};

pub struct Workspace {
    resizer: fir::Resizer,
    blur_temp: Vec<u16>,
    noise_block: Vec<f32>,
    axis_remap: AxisRemapScratch,
    u8_pool: Vec<Vec<u8>>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            resizer: fir::Resizer::new(),
            blur_temp: Vec::new(),
            noise_block: Vec::new(),
            axis_remap: AxisRemapScratch::default(),
            u8_pool: Vec::with_capacity(2),
        }
    }
}

impl Workspace {
    pub(crate) fn take_u8(&mut self, len: usize, clear: bool) -> CoreResult<Vec<u8>> {
        let candidate = self
            .u8_pool
            .iter()
            .position(|buffer| buffer.capacity() >= len)
            .map(|index| self.u8_pool.swap_remove(index));
        let mut buffer = candidate.unwrap_or_default();
        if buffer.capacity() < len {
            buffer
                .try_reserve_exact(len - buffer.len())
                .map_err(|_| CoreError::Runtime("output allocation failed".into()))?;
        }
        buffer.resize(len, 0);
        if clear {
            buffer.fill(0);
        }
        Ok(buffer)
    }

    pub(crate) fn take_staged_u8(
        &mut self,
        len: usize,
        clear: bool,
        reuse: bool,
    ) -> CoreResult<Vec<u8>> {
        if reuse {
            self.take_u8(len, clear)
        } else {
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(len)
                .map_err(|_| CoreError::Runtime("output allocation failed".into()))?;
            buffer.resize(len, 0);
            if clear {
                buffer.fill(0);
            }
            Ok(buffer)
        }
    }

    pub(crate) fn recycle_u8(&mut self, buffer: Vec<u8>) {
        if self.u8_pool.len() < 2 {
            self.u8_pool.push(buffer);
        }
    }

    pub(crate) fn recycle_staged_u8(&mut self, buffer: Vec<u8>, reuse: bool) {
        if reuse {
            self.recycle_u8(buffer);
        }
    }

    pub(crate) fn resizer(&mut self) -> &mut fir::Resizer {
        &mut self.resizer
    }

    pub(crate) fn blur_temp(&mut self) -> &mut Vec<u16> {
        &mut self.blur_temp
    }

    pub(crate) fn noise_block(&mut self) -> &mut Vec<f32> {
        &mut self.noise_block
    }

    pub(crate) fn axis_remap(&mut self) -> &mut AxisRemapScratch {
        &mut self.axis_remap
    }

    pub fn retained_bytes(&self) -> usize {
        let u8_bytes = self
            .u8_pool
            .iter()
            .map(Vec::capacity)
            .fold(0usize, usize::saturating_add);
        u8_bytes
            .saturating_add(
                self.blur_temp
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u16>()),
            )
            .saturating_add(
                self.noise_block
                    .capacity()
                    .saturating_mul(std::mem::size_of::<f32>()),
            )
            .saturating_add(self.axis_remap.retained_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_reuses_capacity_and_retains_at_most_two_buffers() {
        let mut workspace = Workspace::default();
        let first = workspace.take_u8(17, false).unwrap();
        let first_capacity = first.capacity();
        workspace.recycle_u8(first);

        let reused = workspace.take_u8(11, false).unwrap();
        assert_eq!(reused.capacity(), first_capacity);
        workspace.recycle_u8(reused);
        workspace.recycle_u8(vec![0; 23]);
        workspace.recycle_u8(vec![0; 29]);

        assert_eq!(workspace.u8_pool.len(), 2);
    }

    #[test]
    fn fresh_staging_does_not_enter_the_pool() {
        let mut workspace = Workspace::default();
        let buffer = workspace.take_staged_u8(19, true, false).unwrap();
        assert_eq!(buffer, vec![0; 19]);
        workspace.recycle_staged_u8(buffer, false);
        assert!(workspace.u8_pool.is_empty());
    }
}
