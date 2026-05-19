# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
applied-ai/common

Infrastructure used across every volume:
    - llm_client.py: thin wrapper around the model API, logs prompt
      version on every call.
    - tracing.py: OpenTelemetry spans with a graceful no-op fallback.

Everything else lives under its volume and part.
"""

__version__ = "0.1.0"
