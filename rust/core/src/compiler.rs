use crate::{CompiledPipeline, CoreResult, ExecutionMode, PipelineSpec};

pub struct Compiler {
    mode: ExecutionMode,
}

impl Compiler {
    pub fn new(mode: ExecutionMode) -> Self {
        Self { mode }
    }

    pub fn compile(self, spec: PipelineSpec) -> CoreResult<CompiledPipeline> {
        CompiledPipeline::build(spec, self.mode)
    }
}
