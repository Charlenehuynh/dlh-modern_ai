#!/usr/bin/env python3
"""that loads a pre-trained RoBERTa model"""
import transformer

def load_mlm(model_name):
    """return an instance of RobertaForMaskedLM ready for inference."""
    
