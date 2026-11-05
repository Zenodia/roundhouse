# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runs the Ledgerline tasks under the three memory conditions and scores
them. No tools: the model receives the files it needs and returns whole
files (the "patch protocol"), which works the same against a frontier
model through Roundhouse and against a local Dynamo worker, and gives the
long shared prefix the cache measurement needs.
"""
