# Repository Agent Guidance

This repository provides two portable Agent Skills for working with Modules
0-4. Load the relevant `SKILL.md` when the user's request matches it.

Before applying either skill, read `.agents/course-context.md`. It defines the
course scope, important files, ExecuTorch API taxonomy, repository structure,
and the code that should receive the most attention.

## Explain Mode

For requests to learn, explain, review, trace, compare, or test understanding,
read and follow:

`.agents/skills/explain-mode/SKILL.md`

Keep the interaction conversational. Inspect the published course content and corresponding repository
scripts or application code, explain the ExecuTorch APIs, and ask focused
questions that let the learner check and deepen their understanding.

## Build Mode

For requests to build, adapt, debug, extend, or prototype an ExecuTorch model
pipeline or application, read and follow:

`.agents/skills/build-mode/SKILL.md`

Preserve the course's clear separation between model preparation, lowering,
validation, and runtime integration. Explain major ExecuTorch API boundaries
as the implementation progresses. Before adapting a course example into a new
prototype, verify the design against current official ExecuTorch, PyTorch,
platform, and backend documentation.

Treat Modules 0-4 as the course.
