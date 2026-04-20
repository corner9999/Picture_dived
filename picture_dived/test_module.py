#!/usr/bin/env python3
"""Minimal test script for the recognize_image module."""

from __future__ import annotations

import json

from recognize_image import smoke_test

API_KEY = ""


def main() -> int:
    result = smoke_test(
        "imput_picture/test_picture.jpeg",
        api_key=API_KEY,
        max_objects=0,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
