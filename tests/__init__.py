"""Test package.

Makes `tests` a real package so mypy resolves `tests.helpers` under one module
name. Without it, CI's `mypy src scripts tests` sees each test module twice and
refuses to type-check anything.
"""
