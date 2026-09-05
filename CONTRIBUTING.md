# Contributing

Use English for public contributions, including code comments, documentation, commits and discussions. Keep prose paragraphs on one source line and let the reader wrap them.

Use one clear word for directory names unless a tool requires otherwise. Avoid redundant comments, empty scaffolding and dependencies without a concrete use.

## Changes

Open an issue for a bug report or feature request. Small maintenance, documentation and infrastructure changes can go directly to a pull request.

Use type(scope): subject for commit and pull request titles. Keep commit bodies short and focused on the reason for the change.

Pull requests use Why, What and Validation. Link an issue when one exists; an issue is not required for every change.

Record the commands, environment, results and limitations in Validation. When an A/B comparison is needed, include both revisions, controlled conditions, repeat counts and detailed results. Test project behavior without duplicating checks already provided by upstream tools.

## Sign-off

Read the [Developer Certificate of Origin 1.1](https://developercertificate.org/) and sign off contributions you can certify with git commit -s. Contributions use [Apache-2.0](LICENSE). Preserve relevant sign-offs when squashing.

## Publication

Scrub every outgoing contribution before upload: files, the full history being pushed, commit messages, discussions and attachments. Inspect binaries, archives and metadata as well as text for secrets, private content and local paths. Preserve required attribution and genuine sign-off identities.

Record the checked scope, methods and results without exposing sensitive findings. Recheck changed payloads; do not upload artifacts that cannot be adequately inspected.
