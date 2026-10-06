# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains no application code, tests, assets, or build configuration. The directory name is `rag-workout-assistant`; its implementation and architecture are not yet established.

When adding the initial implementation, keep application code, tests, and supporting assets in clearly named directories. Document their actual locations here. Keep related functionality together and avoid speculative layers or unused scaffolding.

## Build, Test, and Development Commands

No build, test, lint, or local run commands are configured yet. Do not assume that commands such as `npm test` or `pytest` are available.

The first implementation should include its dependency manifest and document exact setup, run, and verification commands in a `README.md`. Prefer the chosen ecosystem’s standard tooling and keep commands reproducible from the repository root.

## Coding Style & Naming Conventions

No language or formatter has been selected. Follow the selected language’s standard indentation and naming conventions, and configure its standard formatter when introducing code. Use descriptive names and consistent formatting within each module. Avoid adding dependencies for functionality already provided by the standard library.

## Testing Guidelines

No test framework or coverage threshold exists yet. Add focused, runnable tests for new behavior and bug fixes using the selected stack’s conventional framework. Name tests after the behavior they verify; include invalid input and failure cases where relevant. Document how to run tests before requesting review.

## Commit & Pull Request Guidelines

No Git metadata or commit history is present, so no existing commit convention can be inferred. Use concise, imperative commit subjects, such as `Add workout input validation`.

Pull requests should describe the problem, changes, and verification performed. Link relevant issues when available. Include screenshots for visible UI changes and call out configuration changes or new dependencies.

## Security & Configuration

Never commit credentials, API keys, or private workout records. If configuration is introduced, document required variables and provide an example containing placeholders only. Validate external input at application boundaries.
