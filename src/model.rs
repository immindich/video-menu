// RVM model loading for the NPU; not wired into the daemon yet.
#![allow(dead_code)]

use openvino::{
    CompiledModel, Core, DeviceType, Dimension, InferenceError, Model, PartialShape, RwPropertyKey,
};

fn format_dim(dim: &Dimension) -> String {
    if !dim.is_dynamic() {
        dim.get_min().to_string()
    } else if dim.get_max() < 0 {
        "?".to_string()
    } else {
        format!("{}..{}", dim.get_min(), dim.get_max())
    }
}

fn print_input_shapes(model: &Model) -> Result<(), InferenceError> {
    for i in 0..model.get_inputs_len()? {
        let input = model.get_input_by_index(i)?;
        let name = input.get_name()?;
        let elem_type = input.get_element_type()?;
        let shape = input.get_partial_shape()?;

        let rank = shape.get_rank();
        let dims = if rank.is_dynamic() {
            "<dynamic rank>".to_string()
        } else {
            let dims: Vec<String> = shape.get_dimensions().iter().map(format_dim).collect();
            format!("[{}]", dims.join(", "))
        };

        println!("input {i}: {name} {elem_type} {dims}");
    }
    Ok(())
}

/// Load RVM, fix its input shapes, and compile it for the NPU.
fn compile_model(core: &mut Core, path: &str) -> Result<CompiledModel, InferenceError> {
    // downsample_ratio is baked in as a constant (tools/bake_downsample_ratio.py) so
    // the NPU sees fully static shapes.
    let mut model = core.read_model_from_file(path, "")?;

    // 1280x720 at downsample_ratio 0.375 gives an internal 480x270 resolution;
    // recurrent states are at 1/2, 1/4, 1/8, 1/16 of that (stride-2 convs round up).
    let (h, w) = (270i64, 480i64);
    let down = |n: i64, f: i64| (n + f - 1) / f;
    let shapes = [
        ("src", PartialShape::new_static(4, &[1, 3, 720, 1280])?),
        ("r1i", PartialShape::new_static(4, &[1, 16, down(h, 2), down(w, 2)])?),
        ("r2i", PartialShape::new_static(4, &[1, 20, down(h, 4), down(w, 4)])?),
        ("r3i", PartialShape::new_static(4, &[1, 40, down(h, 8), down(w, 8)])?),
        ("r4i", PartialShape::new_static(4, &[1, 64, down(h, 16), down(w, 16)])?),
    ];
    let shapes: Vec<(&str, &PartialShape)> = shapes.iter().map(|(n, s)| (*n, s)).collect();
    model.reshape(&shapes)?;
    print_input_shapes(&model)?;

    // nixpkgs' openvino doesn't ship the prebuilt plugin compiler
    // (libopenvino_intel_npu_compiler.so), so use the compiler in the NPU driver.
    core.set_property(
        &DeviceType::NPU,
        &RwPropertyKey::Other("NPU_COMPILER_TYPE".into()),
        "DRIVER",
    )?;
    core.compile_model(&model, DeviceType::CPU)
}

