# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Utilities shared by every command; :mod:`poraque.utils.logging` writes their output."""

from .logging import Logger, header, split_blocks

__all__ = ["Logger", "header", "split_blocks"]
