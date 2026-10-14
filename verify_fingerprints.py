"""Recompute the SHA-256 fingerprints of the frozen specification and its addendum.

Each file stores the JSON text followed by a line 'fingerprint: <16 hex characters>'.
The fingerprint is the first 16 characters of the SHA-256 hash of the JSON text.
"""
import hashlib
import pathlib

EXPECTED = {"specification/specification.json": "a1be5ab38d9e20b6",
            "specification/addendum_1.json": "27bc5ad38194a6e9"}

ok = True
for path, expected in EXPECTED.items():
    text = pathlib.Path(path).read_text()
    body, _, stored = text.partition("\n\nfingerprint:")
    computed = hashlib.sha256(body.encode()).hexdigest()[:16]
    match = computed == expected == stored.strip()
    ok &= match
    print(f"{path}: computed {computed}, stored {stored.strip()}, expected {expected} -> {'OK' if match else 'MISMATCH'}")
print("All fingerprints match." if ok else "Fingerprint mismatch!")
