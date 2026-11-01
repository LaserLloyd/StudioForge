# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue.

- **Contact:** use GitHub's private vulnerability reporting: open the
  [Security tab of `LaserLloyd/StudioForge`](https://github.com/LaserLloyd/StudioForge/security)
  and choose **Report a vulnerability**. Only the maintainers can see the report.
- Include the version (`studioforge version`, and `/health` -> `build`), what you did, what you expected
  and what happened. A support bundle (`studioforge support-bundle`) removes credentials, but read it
  before you send it.
- You will get an acknowledgement; the target for a first answer is **7 days**. Please allow
  reasonable time for a fix before you disclose.

## Supported versions

Security fixes go to the latest release line. Once 2.0.0 ships, the previous minor release gets fixes
for 90 days.

## What counts

In scope: authentication or authorization bypass (a key reaching a route above its scope, the bind rule,
the Origin check on MCP), leaking a credential (logs, support bundle, audit log, API responses),
path traversal, SSRF, remote code execution, and anything that lets a network peer or a web page drive
the server without a credential the operator issued.

Out of scope: an attacker who already has administrator rights on the machine, a network you chose to
expose over plain HTTP, denial of service by an authenticated admin, and findings that need physical
access.

How the defences work, and their limits, is in [`docs/SECURITY.md`](docs/SECURITY.md).
