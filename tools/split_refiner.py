"""Cut RVM before its full-resolution stage so the network never sees the
camera resolution.

RVM's refiner is a learned guided filter. Everything learned runs at the
internal (downsampled) resolution and produces per-pixel linear coefficients
A and b (4 channels each: fgr residual RGB + alpha). Only the last step
touches the full-resolution frame `src`:

    x   = concat(src, mean(src, channel))       # R, G, B, gray
    out = upsample(A) * x + upsample(b)         # bilinear, half-pixel
    fgr = clip(out[:3] + src, 0, 1)
    pha = clip(out[3:], 0, 1)

This script removes that step and the initial downsample. The resulting
model takes the already-downsampled frame `src_sm` and outputs `A`, `b` and
the recurrent states. Its shapes depend only on the internal resolution, so
one compiled model serves any camera resolution; the step above is done by
the caller.

usage: python split_refiner.py IN.onnx OUT.onnx
  IN must be the original export (with a `downsample_ratio` input).
"""
import sys

import onnx
from onnx import helper


def find_one(nodes, pred, what):
    found = [n for n in nodes if pred(n)]
    if len(found) != 1:
        sys.exit(f"expected exactly one {what}, found {len(found)}")
    return found[0]


src_path, dst_path = sys.argv[1], sys.argv[2]
model = onnx.load(src_path)
graph = model.graph
nodes = list(graph.node)
producer = {out: n for n in nodes for out in n.output}

# The initial downsample: the Resize applied directly to `src`.
downsample = find_one(
    nodes, lambda n: n.op_type == "Resize" and n.input[0] == "src", "Resize of src"
)
src_sm = downsample.output[0]

# The final step: Add(Mul(Resize(A), x), Resize(b)), where x is built from src.
# Find it structurally from the `fgr` output: fgr = Clip(Add(split0, src)).
fgr_clip = producer["fgr"]
fgr_add = producer[fgr_clip.input[0]]
split = producer[[i for i in fgr_add.input if i != "src"][0]]
affine_add = producer[split.input[0]]
assert affine_add.op_type == "Add", affine_add.op_type
mul = find_one(
    [producer[i] for i in affine_add.input if i in producer],
    lambda n: n.op_type == "Mul",
    "Mul feeding the affine Add",
)
resize_b = producer[[i for i in affine_add.input if i != mul.output[0]][0]]
resize_a = find_one(
    [producer[i] for i in mul.input if i in producer],
    lambda n: n.op_type == "Resize",
    "Resize feeding the affine Mul",
)
assert resize_b.op_type == "Resize", resize_b.op_type
for r in (resize_a, resize_b):
    attrs = {a.name: helper.get_attribute_value(a) for a in r.attribute}
    assert attrs.get("mode") == b"linear", attrs
    assert attrs.get("coordinate_transformation_mode") == b"pytorch_half_pixel", attrs
coef_a, coef_b = resize_a.input[0], resize_b.input[0]

# Rewire: the downsampled frame becomes a graph input.
src_info = find_one(graph.input, lambda i: i.name == "src", "src input")
elem_type = src_info.type.tensor_type.elem_type
for n in nodes:
    for k, name in enumerate(n.input):
        if name == src_sm:
            n.input[k] = "src_sm"
new_inputs = [
    helper.make_tensor_value_info("src_sm", elem_type, ["batch_size", 3, "height", "width"])
] + [i for i in graph.input if i.name.startswith("r") and i.name.endswith("i")]

rec_outputs = [o for o in graph.output if o.name.startswith("r") and o.name.endswith("o")]
new_outputs = [
    helper.make_tensor_value_info("A", elem_type, ["batch_size", 4, "height", "width"]),
    helper.make_tensor_value_info("b", elem_type, ["batch_size", 4, "height", "width"]),
] + rec_outputs

# Rename the coefficient tensors to the output names.
for n in nodes:
    for k, name in enumerate(n.output):
        if name == coef_a:
            n.output[k] = "A"
        elif name == coef_b:
            n.output[k] = "b"
    for k, name in enumerate(n.input):
        if name == coef_a:
            n.input[k] = "A"
        elif name == coef_b:
            n.input[k] = "b"

# Keep only the nodes the new outputs depend on.
producer = {out: n for n in nodes for out in n.output}
needed, stack = set(), [o.name for o in new_outputs]
keep = set()
while stack:
    name = stack.pop()
    if name in needed:
        continue
    needed.add(name)
    if name in producer:
        n = producer[name]
        keep.add(id(n))
        stack.extend(i for i in n.input if i)
kept = [n for n in nodes if id(n) in keep]
if any(i in ("src", "downsample_ratio") for n in kept for i in n.input):
    sys.exit("pruned graph still depends on src or downsample_ratio")

del graph.node[:]
graph.node.extend(kept)
del graph.input[:]
graph.input.extend(new_inputs)
del graph.output[:]
graph.output.extend(new_outputs)
used = {i for n in kept for i in n.input}
inits = [t for t in graph.initializer if t.name in used]
del graph.initializer[:]
graph.initializer.extend(inits)
del graph.value_info[:]

onnx.checker.check_model(model)
onnx.save(model, dst_path)
print(f"wrote {dst_path}: {len(kept)} of {len(nodes)} nodes kept; A={coef_a}, b={coef_b}")
