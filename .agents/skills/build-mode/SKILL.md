---
name: build-mode
description: Build, adapt, debug, or extend ExecuTorch model pipelines and applications using the patterns in the ExecuTorch-on-Arm course repository. Use when a user asks to create an application, change a model or backend, add a deployment target, implement an AOT pipeline, integrate a `.pte` program, or turn a course example into a working prototype.
---

# Build Mode

Build working ExecuTorch software from the repository's patterns while keeping
the model pipeline and runtime integration visible to the user.

Before selecting or adapting a course pattern, read
`.agents/course-context.md`. Use it to identify the relevant entry points,
separate ExecuTorch APIs from supporting framework code, and preserve the
course's AOT and runtime boundaries.

Build mode prioritizes implementation. Recommend the repository's explain-mode
skill when the user wants to learn the course or understand an existing path
in depth. Agents that support named skill invocation can refer to it as
`$explain-mode`. At important milestones, still explain what changed and which
ExecuTorch APIs are involved.

## Define the target

Establish or infer:

- the model and task
- the target platform and hardware
- the backend or portable fallback
- the required input and output contract
- the application language and interface
- the validation target and success criteria

Treat Modules 0-4 as the course reference.

## Select a course pattern

Inspect the closest complete workflow before writing code:

- Module 1 for portable and XNNPACK transformer deployment on Edge Linux
- Module 2 for CNN classification, PTQ, benchmarking, and Python applications
- Module 3 for XNNPACK, Android assets, Java tensors, and mobile lifecycle
- Module 4 for the Arm backend, Ethos-U, TOSA, Vela, FVP, ETDump, and C++

Reuse the architecture and API patterns, not hard-coded model assumptions.
Trace helpers until the real ExecuTorch calls and generated artifacts are
understood.

## Verify against official documentation

Treat the course as a tested example, not the final authority for a new
prototype. Before choosing APIs, packages, backends, build flags, or runtime
integration patterns:

1. Identify the ExecuTorch, PyTorch, TorchAO, backend, and platform versions
   pinned by the relevant module or installed in the user's environment.
2. Read the official documentation listed in the `Official documentation for
   prototypes` section of `.agents/course-context.md`.
3. Check the current export and runtime APIs, API lifecycle state, backend
   requirements, supported operators, quantization flow, package coordinates,
   ABI support, and target integration guidance.
4. Prefer current stable public APIs for new prototypes. Do not carry a private
   binding, compatibility pass, patch, pinned dependency, or course workaround
   into a new application until official documentation or current source shows
   that it is still required.
5. When reproducing a course module, preserve its pinned versions. When moving
   to current releases, describe the required migration rather than silently
   mixing APIs from different versions.
6. If the course and current official documentation differ, explain the
   difference and its likely version boundary before implementing a choice.

Use official ExecuTorch and PyTorch documentation as the primary API source.
For platform and backend details, prefer the official Android, Apple, Arm, or
hardware-vendor documentation. Use blog posts, forum answers, and third-party
examples only as supplementary evidence.

In the implementation handoff, list the official pages that informed important
API, backend, packaging, or runtime decisions.

## Plan the implementation

Keep these responsibilities separate:

1. Prepare weights, labels, preprocessing, and a machine-readable model spec.
2. Export the model with representative example inputs.
3. Quantize only when required, using a backend-compatible quantizer and a
   stated calibration strategy.
4. Lower with the selected partitioner or portable path and write the `.pte`.
5. Inspect delegation and preserve a portable fallback when it is useful.
6. Validate model behavior before integrating it into the application.
7. Load the program, create runtime values, call the exported method, and
   perform application postprocessing on the target.
8. Measure on the target rather than treating host or simulator timings as
   final hardware performance.

For quantized applications, establish an FP32 correctness baseline first, then
deploy INT8 only after checking that its output remains acceptable.

## Implement visibly

- Keep scripts and application code readable enough for a learner to inspect.
- Prefer small helpers with explicit model contracts over hidden automation.
- Name the ExecuTorch APIs used at each boundary.
- Distinguish PyTorch export and PT2E APIs from ExecuTorch backend and runtime
  APIs.
- Preserve preprocessing, layout, shape, dtype, labels, and postprocessing
  consistently between validation and deployment.
- Account for unsupported operations, fallback execution, backend availability,
  memory ownership, threading, and runtime cleanup.
- Do not rewrite unrelated course material or generated artifacts.

Before a substantial edit, summarize the architecture and implementation plan.
After each major slice, report the files changed and the behavior it adds.

## Validate the result

Start with the narrowest useful checks, then expand:

- verify model and metadata paths
- check export and lowering complete
- inspect `.pte` size and delegation
- compare FP32 and quantized outputs when applicable
- exercise preprocessing and output-shape validation
- build or run the target application when the environment permits
- distinguish emulator, FVP, host, and physical-device evidence

Do not claim success when only artifact creation has been checked. Report any
target hardware, dependency, or performance validation that remains.

## Maintain learning checkpoints

At the end of each major phase, briefly explain:

- which ExecuTorch API was used
- what object or artifact entered and left the phase
- why the chosen backend or runtime path fits the target
- what would change with a portable path or another backend

Ask whether the user wants a deeper explanation of that phase. For a complete
walkthrough after implementation, recommend loading
`.agents/skills/explain-mode/SKILL.md`. In clients that support named skill
invocation, suggest:

```text
Use $explain-mode to trace the application we just built from model export to
the final runtime output, and quiz me on the main deployment decisions.
```

Do not let rapid code generation hide unresolved model-contract or deployment
questions from the user.
