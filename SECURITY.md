# Security Policy

## Supported Versions

disdrodb is currently pre-1.0 and includes some changes in every release. As such we can't
guarantee that releases before 1.0 will see security updates except for the most recent
release. After 1.0, you can expect more stability in the interfaces and security fixes to be
backported more regularly.

| Version        | Supported          |
| -------------- | ------------------ |
| 0.x.x (latest) | :white_check_mark: |
| < 0.0.17       | :x:                |

## Unsafe YAML Loading

disdrodb allows for unsafe loading of YAML configuration files. Any YAML files
from untrusted sources should be sanitized of possibly malicious code.

## Reporting a Vulnerability

Do you think you've found a security vulnerability or issue in this project? Let us know by sending
a message to the maintainers. Please include as much information on
the issue as possible like code examples, documentation on the issue in other packages, etc.
