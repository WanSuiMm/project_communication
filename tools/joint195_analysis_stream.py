"""Extract publication-sized slices from the large joint-195 analysis JSON.

The state-pulse effect list can be large. It is consumed one item at a time;
only baseline comparison fields are retained. No model code or third-party
streaming parser is needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


class _JSONStream:
    """Small-buffer JSON reader with recursive skip and selective capture."""

    def __init__(self, stream: TextIO, chunk_size: int = 1 << 20) -> None:
        self.stream = stream
        self.chunk_size = chunk_size
        self.buffer = ""
        self.pos = 0
        self.eof = False

    def _fill(self) -> bool:
        if self.pos:
            self.buffer = self.buffer[self.pos :]
            self.pos = 0
        if self.eof:
            return bool(self.buffer)
        chunk = self.stream.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
            return True
        self.eof = True
        return bool(self.buffer)

    def _peek(self) -> str:
        if self.pos >= len(self.buffer) and not self._fill():
            return ""
        return self.buffer[self.pos]

    def _get(self) -> str:
        ch = self._peek()
        if not ch:
            raise ValueError("Unexpected end of JSON input")
        self.pos += 1
        return ch

    def _space(self) -> None:
        while self._peek() in (" ", "\t", "\r", "\n"):
            self.pos += 1

    def _expect(self, expected: str) -> None:
        self._space()
        actual = self._get()
        if actual != expected:
            raise ValueError(f"Expected {expected!r}, found {actual!r}")

    def _string(self, decode: bool) -> str | None:
        self._space()
        if self._get() != '"':
            raise ValueError("Expected a JSON string")
        chars = ['"'] if decode else None
        while True:
            ch = self._get()
            if chars is not None:
                chars.append(ch)
            if ch == "\\":
                escaped = self._get()
                if chars is not None:
                    chars.append(escaped)
            elif ch == '"':
                break
        if chars is None:
            return None
        value = json.loads("".join(chars))
        if not isinstance(value, str):
            raise ValueError("Invalid JSON string")
        return value

    def _primitive(self, decode: bool) -> Any:
        self._space()
        chars: list[str] = []
        while True:
            ch = self._peek()
            if not ch or ch in ",]} \t\r\n":
                break
            chars.append(self._get())
        if not chars:
            raise ValueError("Expected a JSON primitive")
        if not decode:
            return None
        return json.loads("".join(chars))

    def _skip_value(self) -> None:
        self._space()
        ch = self._peek()
        if ch == '"':
            self._string(False)
        elif ch == "{":
            self._get()
            self._space()
            if self._peek() == "}":
                self._get()
                return
            while True:
                self._string(False)
                self._expect(":")
                self._skip_value()
                self._space()
                separator = self._get()
                if separator == "}":
                    return
                if separator != ",":
                    raise ValueError("Expected ',' or '}' in JSON object")
        elif ch == "[":
            self._get()
            self._space()
            if self._peek() == "]":
                self._get()
                return
            while True:
                self._skip_value()
                self._space()
                separator = self._get()
                if separator == "]":
                    return
                if separator != ",":
                    raise ValueError("Expected ',' or ']' in JSON array")
        else:
            self._primitive(False)

    def _value(self) -> Any:
        self._space()
        ch = self._peek()
        if ch == '"':
            return self._string(True)
        if ch == "{":
            self._get()
            result: dict[str, Any] = {}
            self._space()
            if self._peek() == "}":
                self._get()
                return result
            while True:
                key = self._string(True)
                assert key is not None
                self._expect(":")
                result[key] = self._value()
                self._space()
                separator = self._get()
                if separator == "}":
                    return result
                if separator != ",":
                    raise ValueError("Expected ',' or '}' in JSON object")
        if ch == "[":
            self._get()
            result: list[Any] = []
            self._space()
            if self._peek() == "]":
                self._get()
                return result
            while True:
                result.append(self._value())
                self._space()
                separator = self._get()
                if separator == "]":
                    return result
                if separator != ",":
                    raise ValueError("Expected ',' or ']' in JSON array")
        return self._primitive(True)

    def _cohort_baselines(self) -> dict[str, Any]:
        """Read cohort_effects while discarding all non-baseline values."""
        self._expect("{")
        result: dict[str, Any] = {}
        self._space()
        if self._peek() == "}":
            self._get()
            return result
        while True:
            cohort = self._string(True)
            assert cohort is not None
            self._expect(":")
            self._expect("{")
            fields: dict[str, Any] = {}
            self._space()
            if self._peek() != "}":
                while True:
                    field = self._string(True)
                    assert field is not None
                    self._expect(":")
                    if field == "baseline":
                        fields[field] = self._value()
                    else:
                        self._skip_value()
                    self._space()
                    separator = self._get()
                    if separator == "}":
                        break
                    if separator != ",":
                        raise ValueError("Expected ',' or '}' in cohort record")
            else:
                self._get()
            result[cohort] = fields
            self._space()
            separator = self._get()
            if separator == "}":
                return result
            if separator != ",":
                raise ValueError("Expected ',' or '}' in cohort_effects")

    def _effect(self, bank: str) -> dict[str, Any]:
        """Retain the case binding and baseline side of each effect only."""
        self._expect("{")
        row: dict[str, Any] = {"bank": bank}
        self._space()
        if self._peek() == "}":
            self._get()
            return row
        while True:
            key = self._string(True)
            assert key is not None
            self._expect(":")
            if key in ("case", "baseline_case"):
                row[key] = self._value()
            elif key == "cohort_effects":
                row[key] = self._cohort_baselines()
            else:
                self._skip_value()
            self._space()
            separator = self._get()
            if separator == "}":
                return row
            if separator != ",":
                raise ValueError("Expected ',' or '}' in effect record")

    def _effects(self, bank: str, out: dict[str, dict[str, Any]]) -> None:
        self._expect("[")
        self._space()
        if self._peek() == "]":
            self._get()
            return
        while True:
            row = self._effect(bank)
            case = row.get("case")
            if not isinstance(case, str) or not case:
                raise ValueError(f"Effect item in {bank} has no case name")
            if case in out:
                raise ValueError(f"Duplicate state-pulse case name: {case}")
            out[case] = row
            del row
            self._space()
            separator = self._get()
            if separator == "]":
                return
            if separator != ",":
                raise ValueError("Expected ',' or ']' in effects array")

    def _state_pulses(self, bank: str, out: dict[str, dict[str, Any]]) -> None:
        self._expect("{")
        self._space()
        if self._peek() == "}":
            self._get()
            return
        while True:
            key = self._string(True)
            assert key is not None
            self._expect(":")
            if key == "effects":
                self._effects(bank, out)
            else:
                self._skip_value()
            self._space()
            separator = self._get()
            if separator == "}":
                return
            if separator != ",":
                raise ValueError("Expected ',' or '}' in state_pulse_effects")

    def _bank(self, bank: str, factorial: dict[str, Any],
              matched: dict[str, dict[str, Any]]) -> None:
        self._expect("{")
        self._space()
        if self._peek() == "}":
            self._get()
            return
        while True:
            key = self._string(True)
            assert key is not None
            self._expect(":")
            if key == "factorial":
                factorial[bank] = self._value()
            elif key == "state_pulse_effects":
                self._state_pulses(bank, matched)
            else:
                self._skip_value()
            self._space()
            separator = self._get()
            if separator == "}":
                return
            if separator != ",":
                raise ValueError("Expected ',' or '}' in bank record")

    def _banks(self, factorial: dict[str, Any],
               matched: dict[str, dict[str, Any]]) -> None:
        self._expect("{")
        self._space()
        if self._peek() == "}":
            self._get()
            return
        while True:
            bank = self._string(True)
            assert bank is not None
            self._expect(":")
            self._bank(bank, factorial, matched)
            self._space()
            separator = self._get()
            if separator == "}":
                return
            if separator != ",":
                raise ValueError("Expected ',' or '}' in banks")

    def extract(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "factorial_by_bank": {},
            "confirmation_agreement": None,
            "case_count_check": None,
            "matched_baselines_by_case": {},
        }
        self._expect("{")
        self._space()
        if self._peek() == "}":
            self._get()
        else:
            while True:
                key = self._string(True)
                assert key is not None
                self._expect(":")
                if key == "banks":
                    self._banks(result["factorial_by_bank"], result["matched_baselines_by_case"])
                elif key == "confirmation_agreement":
                    result[key] = self._value()
                elif key == "case_count_check":
                    result[key] = self._value()
                else:
                    self._skip_value()
                self._space()
                separator = self._get()
                if separator == "}":
                    break
                if separator != ",":
                    raise ValueError("Expected ',' or '}' at analysis root")
        self._space()
        if self._peek():
            raise ValueError("Trailing data after top-level JSON object")
        if result["confirmation_agreement"] is None or result["case_count_check"] is None:
            raise ValueError("Required analysis fields are missing")
        if not result["factorial_by_bank"]:
            raise ValueError("No banks.*.factorial records found")
        return result


def extract(path: str | Path) -> dict[str, Any]:
    """Stream the requested summary fields from ``analysis.json``.

    Returns factorial results by bank, the confirmation and case-count
    summaries, and state-pulse baseline comparisons keyed by case name. Each
    effect item is released after its selected baseline fields are retained.
    """
    with Path(path).open("r", encoding="utf-8") as stream:
        return _JSONStream(stream).extract()
