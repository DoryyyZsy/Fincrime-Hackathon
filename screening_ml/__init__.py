"""Screening hit adjudication: SOP-based feature extraction plus a small neural network.

Pipeline: check_data -> (features + silver labels) -> train -> predict -> finetune.
"""
