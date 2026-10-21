"""COPO experiment backends.

This package holds the per-model code for the four COPO experiment models:
LLaVA-1.5 and Qwen-VL-Chat, which are trained through verl's ``main_ppo``
entry point, and InstructBLIP and MiniGPT-4, which are trained through their
own rollout/update loops.  All four are first-class COPO backends selected
with ``--model`` via :mod:`verl.implementations.presets`; a sub-package is
imported only when its model is selected.
"""
