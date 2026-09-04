use crate::capability::{OutputContract, ReadFootprint};
use crate::mask::MaskPlan;
use crate::optimization::{BufferSlot, LoweringPlan};
use crate::plan::TransformPlan;
use crate::{
    BufferExplanation, CopyExplanation, ExecutionMode, ImageContractExplanation,
    PipelineExplanation,
};

pub(crate) fn build(
    transforms: &[TransformPlan],
    mode: ExecutionMode,
    lowering: &LoweringPlan,
    mask: &MaskPlan,
) -> PipelineExplanation {
    let transform_names = transforms.iter().map(TransformPlan::name).collect();
    let mut steps: Vec<_> = transforms.iter().map(TransformPlan::explain).collect();
    for (step, node) in steps.iter_mut().zip(lowering.nodes()) {
        step.input_materialization = node.input.name();
        step.kernel_form = node.kernel.name();
        step.output_slot = node.output.name();
        step.scratch_slots = node.scratch.iter().map(|slot| slot.name()).collect();
        step.selection_reason = node.selection_reason;
        if matches!(
            mode,
            ExecutionMode::StagedFresh | ExecutionMode::StagedReuse
        ) && node.capabilities.read == ReadFootprint::GlobalReduction
            && step.pixel_passes == 2
        {
            step.pixel_passes = 4;
        }
        if step.status == "never" {
            step.execution = "skipped";
            step.pixel_passes = 0;
            step.allocation = "none";
            step.fallback = "none";
            step.input_materialization = "not-required";
            step.kernel_form = "skipped";
            step.output_slot = "unchanged-u8";
            step.scratch_slots.clear();
            step.selection_reason = "probability-zero";
        }
    }

    let passes = steps.iter().filter(|step| step.status != "never").count();
    let pixel_passes = steps.iter().map(|step| step.pixel_passes).sum();
    let normalize_probability =
        transforms
            .iter()
            .zip(lowering.nodes())
            .find_map(|(transform, node)| {
                (node.capabilities.output == OutputContract::TerminalType)
                    .then(|| transform.probability())
            });
    let output_dtype = match normalize_probability {
        Some(1.0) => "float32",
        Some(0.0) | None => "uint8",
        Some(_) => "uint8-or-float32",
    };
    let copy_policy = lowering.copy_policy();
    let direct_input = copy_policy.can_elide;
    let terminal_entry_without_working =
        direct_input && normalize_probability == Some(1.0) && transforms.len() == 1;
    let mut buffers = vec![
        BufferExplanation {
            name: "input",
            dtype: "uint8",
            layout: "HWC",
            lifecycle: "borrowed-for-call",
            condition: "always",
        },
        BufferExplanation {
            name: "working-u8",
            dtype: "uint8",
            layout: "HWC",
            lifecycle: "owned-per-run-workspace-reusable",
            condition: if terminal_entry_without_working {
                "not-required"
            } else if normalize_probability.is_some()
                && transforms.len() == 1
                && copy_policy.count == "0-or-1"
            {
                "sample-dependent"
            } else {
                "always"
            },
        },
    ];
    for (slot, name, dtype, layout, condition) in [
        (
            BufferSlot::ScratchU8,
            "scratch-u8",
            "uint8",
            "HWC",
            "out-of-place-step-applied",
        ),
        (
            BufferSlot::BlurTemp,
            "blur-temp",
            "uint16",
            "HWC",
            "GaussianBlur-applied",
        ),
        (
            BufferSlot::NoiseBlock,
            "noise-f32-block",
            "float32",
            "block",
            "GaussianNoise-applied",
        ),
        (
            BufferSlot::AxisRemap,
            "axis-remap",
            "float32",
            "axes",
            "GridDistortion-applied",
        ),
    ] {
        if lowering.uses_effective_slot(transforms, slot) {
            buffers.push(BufferExplanation {
                name,
                dtype,
                layout,
                lifecycle: "owned-per-run-workspace-reusable",
                condition,
            });
        }
    }
    if normalize_probability.is_some_and(|p| p > 0.0) {
        buffers.push(BufferExplanation {
            name: "output-f32",
            dtype: "float32",
            layout: "HWC",
            lifecycle: "owned-output",
            condition: "Normalize-applied",
        });
    }
    let mut fallbacks: Vec<_> = steps.iter().map(|step| step.fallback).collect();
    fallbacks.retain(|fallback| *fallback != "none");
    fallbacks.sort_unstable();
    fallbacks.dedup();

    PipelineExplanation {
        mode: match mode {
            ExecutionMode::Reference => "reference",
            ExecutionMode::Compiled => "compiled",
            ExecutionMode::StagedFresh => "staged-fresh",
            ExecutionMode::StagedReuse => "staged-reuse",
        },
        sampling: "native-plan-before-execution",
        transforms: transform_names,
        steps,
        fusions: Vec::new(),
        unit_specializations: lowering
            .nodes()
            .iter()
            .zip(transforms)
            .filter_map(|(node, transform)| {
                (transform.probability() > 0.0)
                    .then_some(node.unit_specialization)
                    .flatten()
            })
            .collect(),
        optimizations: direct_input
            .then_some("input-copy-elision")
            .into_iter()
            .collect(),
        passes,
        pixel_passes,
        output_dtype,
        output_layout: "HWC",
        input: ImageContractExplanation {
            container: "borrowed-buffer",
            dtype: "uint8",
            layout: "HWC",
            channels: "RGB",
            contiguous: true,
            ownership: "caller",
        },
        output: ImageContractExplanation {
            container: "owned-buffer",
            dtype: output_dtype,
            layout: "HWC",
            channels: "RGB",
            contiguous: true,
            ownership: "result",
        },
        buffers,
        copies: vec![CopyExplanation {
            stage: "native-entry",
            count: copy_policy.count,
            condition: copy_policy.condition,
            reason: "establish-owned-working-buffer",
        }],
        fallbacks,
        mask_plan: mask.explain(0),
    }
}
