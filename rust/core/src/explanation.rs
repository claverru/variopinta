use crate::capability::{OutputContract, ReadFootprint};
use crate::optimization::{BufferSlot, LoweringPlan};
use crate::plan::{owned_simd_fallback, TransformPlan};
use crate::{
    BufferExplanation, CopyExplanation, ExecutionMode, ImageContractExplanation,
    PipelineExplanation,
};

pub(crate) fn build(
    transforms: &[TransformPlan],
    mode: ExecutionMode,
    lowering: &LoweringPlan,
) -> PipelineExplanation {
    let transform_names = transforms.iter().map(TransformPlan::name).collect();
    let mut steps: Vec<_> = transforms.iter().map(TransformPlan::explain).collect();
    for (step, node) in steps.iter_mut().zip(lowering.nodes()) {
        step.input_materialization = node.input.name();
        step.kernel_form = node.kernel.name();
        step.output_slot = node.output.name();
        step.scratch_slots = node.scratch.iter().map(|slot| slot.name()).collect();
        step.selection_reason = node.selection_reason;
    }
    let fusion = lowering.fusion();
    let normalize_to_torch_fused = fusion.is_some_and(|selection| selection.is_active());
    if matches!(
        mode,
        ExecutionMode::StagedFresh | ExecutionMode::StagedReuse
    ) {
        for (step, node) in steps.iter_mut().zip(lowering.nodes()) {
            if node.capabilities.read == ReadFootprint::GlobalReduction && step.pixel_passes == 2 {
                step.pixel_passes = 4;
            }
        }
    }
    for (index, pair) in lowering.nodes().windows(2).enumerate() {
        if pair[0].capabilities.output == OutputContract::TerminalType
            && pair[1].capabilities.output == OutputContract::TerminalLayout
        {
            steps[index].execution = "out-of-place";
        }
    }
    if let Some(selection) = fusion.filter(|_| normalize_to_torch_fused) {
        let execution = if selection.probability == 1.0 {
            "fused-terminal"
        } else {
            "conditional-fused-terminal"
        };
        steps[selection.first].execution = execution;
        steps[selection.second].execution = execution;
        steps[selection.second].pixel_passes = 0;
        steps[selection.second].fallback = owned_simd_fallback();
    }
    for step in &mut steps {
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
    let to_torch = lowering
        .nodes()
        .last()
        .is_some_and(|node| node.capabilities.output == OutputContract::TerminalLayout);
    let normalize_to_torch_probability = fusion.map(|selection| selection.probability);
    let output_dtype = match normalize_probability {
        Some(1.0) => "float32",
        Some(0.0) | None => "uint8",
        Some(_) => "uint8-or-float32",
    };
    let copy_policy = lowering.copy_policy();
    let direct_input = copy_policy.can_elide;
    let terminal_entry_without_working = direct_input
        && (fusion.is_some_and(|selection| selection.first == 0)
            || (lowering.nodes().len() == 1
                && lowering.nodes()[0].capabilities.output == OutputContract::TerminalLayout)
            || (normalize_probability == Some(1.0) && transforms.len() == 1));
    let mut buffers = vec![BufferExplanation {
        name: "input",
        dtype: "uint8",
        layout: "HWC",
        lifecycle: "borrowed-for-call",
        condition: "always",
    }];
    buffers.push(BufferExplanation {
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
    });
    if lowering.uses_effective_slot(transforms, BufferSlot::ScratchU8) {
        buffers.push(BufferExplanation {
            name: "scratch-u8",
            dtype: "uint8",
            layout: "HWC",
            lifecycle: "owned-per-run-workspace-reusable",
            condition: "out-of-place-step-applied",
        });
    }
    if lowering.uses_effective_slot(transforms, BufferSlot::BlurTemp) {
        buffers.push(BufferExplanation {
            name: "blur-temp",
            dtype: "uint16",
            layout: "HWC",
            lifecycle: "owned-per-run-workspace-reusable",
            condition: "GaussianBlur-applied",
        });
    }
    if to_torch && normalize_probability.is_some_and(|p| p > 0.0) && !normalize_to_torch_fused {
        buffers.push(BufferExplanation {
            name: "normalized-f32",
            dtype: "float32",
            layout: "HWC",
            lifecycle: "owned-per-run-intermediate",
            condition: "Normalize-applied",
        });
    }
    if to_torch && normalize_probability != Some(1.0) {
        buffers.push(BufferExplanation {
            name: "output-u8",
            dtype: "uint8",
            layout: "CHW",
            lifecycle: "owned-output",
            condition: if normalize_probability.is_some() {
                "Normalize-skipped"
            } else {
                "always"
            },
        });
    }
    if normalize_probability.is_some_and(|p| p > 0.0) {
        buffers.push(BufferExplanation {
            name: "output-f32",
            dtype: "float32",
            layout: if to_torch { "CHW" } else { "HWC" },
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
        fusions: fusion
            .filter(|selection| selection.is_active())
            .map(|selection| selection.rule.name())
            .into_iter()
            .collect(),
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
        transforms: transform_names,
        steps,
        output_dtype,
        output_layout: if to_torch { "CHW" } else { "HWC" },
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
            layout: if to_torch { "CHW" } else { "HWC" },
            channels: "RGB",
            contiguous: true,
            ownership: "result",
        },
        buffers,
        copies: std::iter::once(CopyExplanation {
            stage: "native-entry",
            count: copy_policy.count,
            condition: copy_policy.condition,
            reason: "establish-owned-working-buffer",
        })
        .chain(to_torch.then_some(CopyExplanation {
            stage: "terminal-layout",
            count: match normalize_to_torch_probability {
                Some(1.0) if normalize_to_torch_fused => "0",
                Some(0.0) | None => "1",
                Some(_) if normalize_to_torch_fused => "0-or-1",
                Some(_) => "1",
            },
            condition: match normalize_to_torch_probability {
                Some(1.0) if normalize_to_torch_fused => "always-fused",
                Some(0.0) | None => "always",
                Some(_) if normalize_to_torch_fused => "sample-dependent",
                Some(_) => "always",
            },
            reason: if normalize_to_torch_fused {
                "normalization-writes-contiguous-CHW"
            } else {
                "HWC-to-contiguous-CHW"
            },
        }))
        .collect(),
        fallbacks,
    }
}
