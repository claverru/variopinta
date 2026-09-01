# Security policy

Variopinta is experimental. Security fixes are provided for the latest patch of
the current `0.1` release line.

| Version | Supported |
|---|---|
| Latest `0.1.x` | Yes |
| Earlier releases | No |

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Report it privately
through [GitHub Security Advisories](https://github.com/claverru/variopinta/security/advisories/new).

Include the affected revision or version, platform, Python version, input or
pipeline needed to reproduce the issue, observed impact, and any relevant crash
or sanitizer output. Avoid including sensitive data that is not needed to
reproduce the report.

Relevant reports include memory-safety defects, unchecked allocation or
dimension handling, malformed-image issues, path or file-handling defects, and
unexpected execution of Python or native code.
