# Password Hash Verification (Python)

A simple Python example demonstrating how to hash a password and securely verify it using the `crypt` module and constant-time comparison.

## What This Code Does

- Hashes a plaintext password using `crypt`
- Re-hashes the input password using the stored hash as salt
- Compares hashes securely using `hmac.compare_digest`

## Technologies Used

Python 3, crypt module, hmac module
