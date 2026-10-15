# Security Policy

## Supported versions

Only the latest release receives security fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report them privately through GitHub security advisories:
https://github.com/jere344/vinted-hunter/security/advisories/new

Include:

- A description of the issue and its impact.
- Steps to reproduce, or a proof of concept.
- The version or commit you tested.

You can expect an acknowledgement within a few days. Fixes are released as soon as they are verified,
and the reporter is credited unless they prefer otherwise.

## Scope notes

Vinted Hunter runs locally and binds to `127.0.0.1` by default. Do not expose it to a network without
adding authentication. Keep API keys in `.env`, which is git-ignored, and never paste them into issues.
