# Security Notes

This project is a computer vision prototype and should not be treated as a complete authentication system.

Before using it for authentication or access control, add:

- liveness detection and anti-spoofing
- encrypted storage for trained encodings
- protected enrollment and retraining workflows
- rate limiting and lockout behavior
- audit logging
- target-platform security review

Keep generated datasets, trained encoding files, logs, and environment-specific configuration outside version control.
