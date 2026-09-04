use crate::capability::{
    ExecutionForm, OutputContract, ReusableScratchSlot, ScratchRequirement, TransformCapabilities,
    WriteCoverage,
};
use crate::kernels::{self, KernelImplementation};
use crate::plan::{SampledTransform, TransformPlan};
use crate::{CoreError, CoreResult, ExecutionMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputMaterialization {
    BorrowedInput,
    OwnedInputCopy,
    PreviousOutput,
}

impl InputMaterialization {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::BorrowedInput => "borrowed-input",
            Self::OwnedInputCopy => "owned-input-copy",
            Self::PreviousOutput => "previous-output",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BufferSlot {
    WorkingU8,
    ScratchU8,
    BlurTemp,
    NoiseBlock,
    AxisRemap,
    OutputF32Hwc,
}

impl BufferSlot {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::WorkingU8 => "working-u8",
            Self::ScratchU8 => "scratch-u8",
            Self::BlurTemp => "blur-temp",
            Self::NoiseBlock => "noise-f32-block",
            Self::AxisRemap => "axis-remap",
            Self::OutputF32Hwc => "output-f32-hwc",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputRoute {
    Always(BufferSlot),
    NormalizeHwcConditional,
}

impl OutputRoute {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Always(slot) => slot.name(),
            Self::NormalizeHwcConditional => "output-f32-hwc-or-working-u8",
        }
    }

    fn uses(self, slot: BufferSlot) -> bool {
        match self {
            Self::Always(actual) => actual == slot,
            Self::NormalizeHwcConditional => {
                matches!(slot, BufferSlot::OutputF32Hwc | BufferSlot::WorkingU8)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelSelection {
    Form(ExecutionForm),
}

impl KernelSelection {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Form(ExecutionForm::BorrowedToOwned) => "borrowed-to-owned",
            Self::Form(ExecutionForm::OwnedToOwned) => "owned-to-owned",
            Self::Form(ExecutionForm::OwnedInPlace) => "owned-in-place",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LoweringNode {
    pub capabilities: TransformCapabilities,
    pub input: InputMaterialization,
    pub kernel: KernelSelection,
    pub output: OutputRoute,
    pub scratch: Vec<BufferSlot>,
    pub unit_specialization: Option<&'static str>,
    pub selection_reason: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CopyPolicy {
    pub count: &'static str,
    pub condition: &'static str,
    pub can_elide: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LoweringPlan {
    nodes: Vec<LoweringNode>,
    entry_prerequisites: usize,
    copy_policy: CopyPolicy,
}

impl LoweringPlan {
    pub(crate) fn compile(transforms: &[TransformPlan], mode: ExecutionMode) -> CoreResult<Self> {
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(transforms.len())
            .map_err(|_| CoreError::Runtime("lowering plan allocation failed".into()))?;

        for (index, transform) in transforms.iter().enumerate() {
            let capabilities = transform.capabilities();
            let implementations = kernels::implementations(transform);
            validate_implementations(capabilities, implementations)?;

            let borrowed_entry = mode == ExecutionMode::Compiled
                && index == 0
                && matches!(
                    capabilities.write,
                    WriteCoverage::FullOverwrite | WriteCoverage::TerminalConversion
                );
            let implementation = select_implementation(transform, implementations, borrowed_entry)
                .ok_or_else(|| {
                    CoreError::Runtime(format!(
                        "no selected kernel implementation for {}",
                        transform.name()
                    ))
                })?;
            let form = implementation.form;
            let input = match form {
                ExecutionForm::BorrowedToOwned => InputMaterialization::BorrowedInput,
                ExecutionForm::OwnedToOwned | ExecutionForm::OwnedInPlace if index == 0 => {
                    InputMaterialization::OwnedInputCopy
                }
                ExecutionForm::OwnedToOwned | ExecutionForm::OwnedInPlace => {
                    InputMaterialization::PreviousOutput
                }
            };
            let output = output_route(transforms, index, capabilities.output, form);
            let mut scratch = scratch_slots(capabilities.scratch);
            scratch.sort_unstable_by_key(|slot| slot.name());
            scratch.dedup();
            let unit_specialization = if matches!(
                mode,
                ExecutionMode::StagedFresh | ExecutionMode::StagedReuse
            ) {
                None
            } else {
                implementation.unit_specialization
            };
            nodes.push(LoweringNode {
                capabilities,
                input,
                kernel: KernelSelection::Form(form),
                output,
                scratch,
                unit_specialization,
                selection_reason: selection_reason(transform, implementation),
            });
        }

        let entry_prerequisites = entry_prerequisites(&nodes);
        let copy_policy = copy_policy(transforms, &nodes, entry_prerequisites, mode);
        Ok(Self {
            nodes,
            entry_prerequisites,
            copy_policy,
        })
    }

    pub(crate) fn nodes(&self) -> &[LoweringNode] {
        &self.nodes
    }

    pub(crate) fn node(&self, index: usize) -> &LoweringNode {
        &self.nodes[index]
    }

    pub(crate) fn copy_policy(&self) -> CopyPolicy {
        self.copy_policy
    }

    pub(crate) fn uses_effective_slot(
        &self,
        transforms: &[TransformPlan],
        slot: BufferSlot,
    ) -> bool {
        transforms.iter().zip(&self.nodes).any(|(transform, node)| {
            transform.probability() > 0.0
                && (node.output.uses(slot) || node.scratch.contains(&slot))
        })
    }

    pub(crate) fn entry_ready(&self, sampled: &[SampledTransform]) -> bool {
        self.entry_prerequisites > 0
            && sampled
                .iter()
                .take(self.entry_prerequisites)
                .all(|sample| !matches!(sample, SampledTransform::Skip))
    }
}

fn validate_implementations(
    capabilities: TransformCapabilities,
    implementations: &[KernelImplementation],
) -> CoreResult<()> {
    if implementations.is_empty()
        || implementations
            .iter()
            .any(|implementation| !capabilities.legal_forms.contains(&implementation.form))
    {
        return Err(CoreError::Runtime(
            "kernel catalog contradicts transform capabilities".into(),
        ));
    }
    Ok(())
}

fn preferred_implementation(
    implementations: &[KernelImplementation],
    form: ExecutionForm,
) -> Option<KernelImplementation> {
    implementations
        .iter()
        .copied()
        .find(|implementation| implementation.form == form)
}

fn preferred_owned_implementation(
    implementations: &[KernelImplementation],
) -> Option<KernelImplementation> {
    preferred_implementation(implementations, ExecutionForm::OwnedInPlace)
        .or_else(|| preferred_implementation(implementations, ExecutionForm::OwnedToOwned))
}

fn select_implementation(
    transform: &TransformPlan,
    implementations: &[KernelImplementation],
    borrowed_entry: bool,
) -> Option<KernelImplementation> {
    if matches!(
        transform,
        TransformPlan::Affine { .. } | TransformPlan::RandomRotation { .. }
    ) {
        return preferred_implementation(implementations, ExecutionForm::OwnedToOwned);
    }
    if borrowed_entry {
        preferred_implementation(implementations, ExecutionForm::BorrowedToOwned)
            .or_else(|| preferred_owned_implementation(implementations))
    } else {
        preferred_owned_implementation(implementations)
    }
}

fn selection_reason(
    transform: &TransformPlan,
    implementation: KernelImplementation,
) -> &'static str {
    if matches!(
        transform,
        TransformPlan::Affine { .. } | TransformPlan::RandomRotation { .. }
    ) && implementation.form == ExecutionForm::OwnedToOwned
    {
        "benchmark-policy:affine-copy-then-transform"
    } else if implementation.unit_specialization.is_some() {
        "benchmark-policy:composed-color-matrix"
    } else {
        match implementation.form {
            ExecutionForm::BorrowedToOwned => "benchmark-policy:elide-entry-copy",
            ExecutionForm::OwnedToOwned => "semantic-legality:owned-to-owned",
            ExecutionForm::OwnedInPlace => "semantic-legality:owned-in-place",
        }
    }
}

fn output_route(
    transforms: &[TransformPlan],
    index: usize,
    output: OutputContract,
    form: ExecutionForm,
) -> OutputRoute {
    let slot = match output {
        OutputContract::TerminalType => {
            let probability = transforms[index].probability();
            return probability_route(
                probability,
                BufferSlot::OutputF32Hwc,
                BufferSlot::WorkingU8,
                OutputRoute::NormalizeHwcConditional,
            );
        }
        OutputContract::ShapePreserving
        | OutputContract::StaticallySized
        | OutputContract::SampleSized => match form {
            ExecutionForm::OwnedToOwned => BufferSlot::ScratchU8,
            ExecutionForm::BorrowedToOwned | ExecutionForm::OwnedInPlace => BufferSlot::WorkingU8,
        },
    };
    OutputRoute::Always(slot)
}

fn probability_route(
    probability: f32,
    applied: BufferSlot,
    skipped: BufferSlot,
    conditional: OutputRoute,
) -> OutputRoute {
    if probability == 1.0 {
        OutputRoute::Always(applied)
    } else if probability == 0.0 {
        OutputRoute::Always(skipped)
    } else {
        conditional
    }
}

fn scratch_slots(contract: ScratchRequirement) -> Vec<BufferSlot> {
    match contract {
        ScratchRequirement::None | ScratchRequirement::Stack { .. } => Vec::new(),
        ScratchRequirement::Reusable {
            slot: ReusableScratchSlot::U8,
            ..
        } => vec![BufferSlot::ScratchU8],
        ScratchRequirement::Reusable {
            slot: ReusableScratchSlot::U16,
            ..
        } => vec![BufferSlot::BlurTemp],
        ScratchRequirement::Reusable {
            slot: ReusableScratchSlot::NoiseBlock,
            ..
        } => vec![BufferSlot::NoiseBlock],
        ScratchRequirement::Reusable {
            slot: ReusableScratchSlot::AxisRemap,
            ..
        } => vec![BufferSlot::AxisRemap],
    }
}

fn entry_prerequisites(nodes: &[LoweringNode]) -> usize {
    if !matches!(
        nodes.first().map(|node| node.kernel),
        Some(KernelSelection::Form(ExecutionForm::BorrowedToOwned))
    ) {
        return 0;
    }
    1
}

fn copy_policy(
    transforms: &[TransformPlan],
    nodes: &[LoweringNode],
    entry_prerequisites: usize,
    mode: ExecutionMode,
) -> CopyPolicy {
    if mode != ExecutionMode::Compiled || nodes.is_empty() || entry_prerequisites == 0 {
        return CopyPolicy {
            count: "1",
            condition: "always",
            can_elide: false,
        };
    }
    probability_policy(
        transforms
            .iter()
            .take(entry_prerequisites)
            .map(TransformPlan::probability),
    )
}

fn probability_policy(probabilities: impl Iterator<Item = f32>) -> CopyPolicy {
    let mut all_certain = true;
    let mut any_never = false;
    for probability in probabilities {
        all_certain &= probability == 1.0;
        any_never |= probability == 0.0;
    }
    if all_certain {
        CopyPolicy {
            count: "0",
            condition: "always-elided",
            can_elide: true,
        }
    } else if any_never {
        CopyPolicy {
            count: "1",
            condition: "always",
            can_elide: false,
        }
    } else {
        CopyPolicy {
            count: "0-or-1",
            condition: "sample-dependent",
            can_elide: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BorderMode, Interpolation, TransformSpec};

    fn plans(specs: Vec<TransformSpec>) -> Vec<TransformPlan> {
        TransformPlan::compile(specs).unwrap()
    }

    #[test]
    fn copy_policy_is_derived_from_selected_lowerings() {
        let transforms = plans(vec![TransformSpec::Resize {
            height: 7,
            width: 11,
            interpolation: Interpolation::Bilinear,
            antialias: false,
            p: 0.5,
        }]);
        let lowering = LoweringPlan::compile(&transforms, ExecutionMode::Compiled).unwrap();
        assert_eq!(
            lowering.node(0).kernel,
            KernelSelection::Form(ExecutionForm::BorrowedToOwned)
        );
        assert_eq!(lowering.copy_policy().count, "0-or-1");

        let reference = LoweringPlan::compile(&transforms, ExecutionMode::Reference).unwrap();
        assert_eq!(
            reference.node(0).input,
            InputMaterialization::OwnedInputCopy
        );
        assert!(!reference.copy_policy().can_elide);
    }

    #[test]
    fn affine_keeps_the_evidence_selected_owned_lowering() {
        let transforms = plans(vec![TransformSpec::Affine {
            degrees: [-10.0, 10.0],
            translate: [0.0, 0.0],
            scale: [1.0, 1.0],
            shear: [0.0; 4],
            interpolation: Interpolation::Bilinear,
            border_mode: BorderMode::Constant,
            fill: [0; 3],
            p: 1.0,
        }]);
        let lowering = LoweringPlan::compile(&transforms, ExecutionMode::Compiled).unwrap();
        assert_eq!(
            lowering.node(0).kernel,
            KernelSelection::Form(ExecutionForm::OwnedToOwned)
        );
        assert_eq!(lowering.copy_policy().count, "1");
        assert_eq!(
            lowering.node(0).selection_reason,
            "benchmark-policy:affine-copy-then-transform"
        );
    }

    #[test]
    fn crop_entry_depends_only_on_the_crop_node() {
        let transforms = plans(vec![
            TransformSpec::CenterCrop {
                height: 7,
                width: 11,
                p: 1.0,
            },
            TransformSpec::Resize {
                height: 5,
                width: 9,
                interpolation: Interpolation::Bilinear,
                antialias: false,
                p: 0.0,
            },
        ]);
        let lowering = LoweringPlan::compile(&transforms, ExecutionMode::Compiled).unwrap();
        assert_eq!(lowering.copy_policy().count, "0");
        assert!(lowering.entry_ready(&[
            SampledTransform::CenterCrop(crate::plan::CropSample {
                top: 0,
                left: 0,
                height: 7,
                width: 11,
            }),
            SampledTransform::Skip,
        ]));
    }

    #[test]
    fn every_catalog_transform_has_capabilities_and_a_legal_kernel() {
        for spec in crate::catalog::valid_fixtures() {
            let transforms = plans(vec![spec]);
            LoweringPlan::compile(&transforms, ExecutionMode::Compiled).unwrap();
        }
    }
}
