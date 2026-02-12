# SAE Investigation

## Overview
General-purpose SAE-based interpretability investigation using Gemma Scope.
Uses the `sae_tools` library for high-level SAE operations.

## Setup
- Model: Gemma 2 9B IT
- GPU: A100
- Tools: sae_lens, Neuronpedia API, sae_tools library

## Available Functions
- `analyze_prompt()` - Find top SAE features for a prompt
- `compare_prompts()` - Differential feature analysis between prompts
- `search_features()` - Search Neuronpedia for features by description
- `steer_with_feature()` - Steer generation using SAE feature directions
- `load_sae()` - Load and cache Gemma Scope SAEs
