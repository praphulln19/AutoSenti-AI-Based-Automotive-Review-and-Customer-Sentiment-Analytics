"""
src/data/validator.py
──────────────────────
Input validation layer for the single-review inference workflow.

Responsibilities
────────────────
- Validate single review text for inference.
Returns structured ValidationResult objects so the UI can render
user-friendly messages without exposing internal exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def summary(self) -> str:
        parts = []
        if self.errors:
            parts.append("Errors:\n" + "\n".join(f"  • {e}" for e in self.errors))
        if self.warnings:
            parts.append("Warnings:\n" + "\n".join(f"  • {w}" for w in self.warnings))
        return "\n".join(parts) if parts else "Validation passed."


# ── Single review validation ──────────────────────────────────────────────────

def validate_review_text(text: str,
                          min_length: int = 5,
                          max_length: int = 1000) -> ValidationResult:
    """Validate a single review string submitted for inference."""
    result = ValidationResult(is_valid=True)

    if not isinstance(text, str):
        result.add_error("Review must be a text string.")
        return result

    stripped = text.strip()

    if not stripped:
        result.add_error("Review text cannot be empty.")
        return result

    if len(stripped) < min_length:
        result.add_error(
            f"Review is too short ({len(stripped)} chars). "
            f"Please provide at least {min_length} characters."
        )

    if len(stripped) > max_length:
        result.add_warning(
            f"Review is very long ({len(stripped)} chars). "
            f"Only the first {max_length} characters will be analysed."
        )

    return result


