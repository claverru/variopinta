use augment_core::{
    Compiler, ExecutionMode, Interpolation, PipelineOutput, PipelineSpec, TargetBuffer,
    TargetInput, TargetOutput, TargetRequirements, TargetSpec, TransformSpec, Workspace,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn allocated(size: usize) {
    ALLOCATIONS.fetch_add(1, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: all allocation operations forward the original pointer and layout to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc supplies a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout are the unchanged allocation pair.
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: GlobalAlloc supplies the original allocation and valid new size.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let size: usize = args[1].parse().unwrap();
    let case = &args[2];
    let expanded = args[3] == "rgb-expansion";
    let out = size * 3 / 4;
    let resize = TransformSpec::Resize {
        height: out,
        width: out,
        interpolation: Interpolation::Bilinear,
        antialias: false,
        p: 1.0,
    };
    let transforms = match case.as_str() {
        "geometry" => vec![
            TransformSpec::CenterCrop {
                height: size - 2,
                width: size - 2,
                p: 1.0,
            },
            resize,
        ],
        "filtering" => vec![
            TransformSpec::GaussianBlur {
                kernel_size: 5,
                sigma: [1.1; 2],
                p: 1.0,
            },
            TransformSpec::Sharpen {
                alpha: [0.5; 2],
                lightness: [1.0; 2],
                p: 1.0,
            },
        ],
        "normalized-tensor" => vec![
            resize,
            TransformSpec::Normalize {
                mean: vec![0.5],
                std: vec![0.5],
                max_pixel_value: 255.0,
                p: 1.0,
            },
        ],
        _ => panic!("unknown case"),
    };
    let requirements = if case == "normalized-tensor" {
        TargetRequirements::CHW
    } else {
        TargetRequirements::HWC
    };
    let plan = Compiler::new(ExecutionMode::Compiled)
        .compile(PipelineSpec::with_target_requirements(
            transforms,
            vec![(TargetSpec::Image, requirements)],
        ))
        .unwrap();
    let source = vec![73; size * size];
    let mut workspace = Workspace::default();
    let run = |workspace: &mut Workspace| {
        let rgb;
        let data = if expanded {
            rgb = source
                .iter()
                .flat_map(|&pixel| [pixel; 3])
                .collect::<Vec<_>>();
            &rgb
        } else {
            &source
        };
        let input = TargetInput {
            role: TargetSpec::Image,
            data: TargetBuffer::Borrowed(data),
            height: size,
            width: size,
            channels: if expanded { 3 } else { 1 },
            rank: if expanded { 3 } else { 2 },
        };
        let mut outputs = plan.apply_targets(vec![input], 137, 7, workspace).unwrap();
        let TargetOutput::Image(mut image) = outputs.pop().unwrap() else {
            panic!()
        };
        let output = image.hwc.take().or(image.chw.take()).unwrap();
        if !expanded {
            return output;
        }
        match output {
            PipelineOutput::U8Hwc {
                data,
                height,
                width,
                ..
            } => PipelineOutput::U8Hwc {
                data: data.chunks_exact(3).map(|p| p[0]).collect(),
                height,
                width,
                channels: 1,
                rank: 2,
            },
            PipelineOutput::F32Chw {
                data,
                height,
                width,
                ..
            } => PipelineOutput::F32Chw {
                data: data[..height * width].to_vec(),
                height,
                width,
                channels: 1,
                rank: 3,
            },
            _ => panic!(),
        }
    };
    for phase in ["cold", "warm"] {
        let baseline = LIVE.load(Relaxed);
        PEAK.store(baseline, Relaxed);
        let allocations = ALLOCATIONS.load(Relaxed);
        let output = run(&mut workspace);
        let count = ALLOCATIONS.load(Relaxed) - allocations;
        let peak = PEAK.load(Relaxed).saturating_sub(baseline);
        let retained = workspace.retained_bytes();
        println!("{{\"phase\":\"{phase}\",\"allocations\":{count},\"peak_additional_live_bytes\":{peak},\"retained_workspace_bytes\":{retained}}}");
        drop(output);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn allocator_preserves_bytes_across_growth() {
        let mut data = vec![73u8; 37];
        data.reserve(8192);
        assert_eq!(&data[..], &[73; 37]);
        data.resize(8192, 19);
        data.shrink_to_fit();
        assert!(data[37..].iter().all(|&v| v == 19));
        assert_eq!(&data[..37], &[73; 37]);
    }
}
