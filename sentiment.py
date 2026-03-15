"""
Natural Language Processing (NLP) for Sentiment Analysis.

Uses the ProsusAI/finbert model (fine-tuned on financial texts) to parse
headlines and calculate an aggregated sentiment score from -1.0 to 1.0.
"""

from __future__ import annotations

import os
# Disable tokenizer parallelism to avoid macOS deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force transformers to avoid TensorFlow import path (can hang on macOS in threaded workers)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import logging
from typing import List

log = logging.getLogger("sentiment")

class FinBERTSentiment:
    _instance = None

    def __init__(self):
        # We use a singleton pattern or lazy loading to avoid loading 
        # the ML model into memory multiple times.
        if FinBERTSentiment._instance is not None:
            self.pipeline = FinBERTSentiment._instance.pipeline
        else:
            log.info("Loading FinBERT sentiment model (this may take a moment)...")
            try:
                from transformers import pipeline
                import torch
            except ImportError:
                log.error("Missing NLP dependencies. Please run: pip install transformers torch")
                self.pipeline = None
                FinBERTSentiment._instance = self
                return

            # Keep torch threading conservative inside background worker threads.
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            
            # Force CPU by default to avoid MPS (Metal) deadlocks on Mac
            device = -1
            
            self.pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
            FinBERTSentiment._instance = self

    def score_headlines(self, headlines: List[str]) -> float:
        """
        Run a list of headlines through FinBERT and return an aggregated daily sentiment score.
        Score ranges from -1.0 to 1.0.
        """
        if not headlines or self.pipeline is None:
            return 0.0
            
        try:
            results = self.pipeline(headlines)
        except Exception as e:
            log.error("FinBERT model evaluation failed: %s", e)
            return 0.0
            
        scores = []
        for res in results:
            label = res["label"].lower()
            score = res["score"]
            # Convert pipeline output into continuous -1 to 1 score
            if label == "positive":
                scores.append(score)
            elif label == "negative":
                scores.append(-score)
            else:
                scores.append(0.0) # Neutral
                
        # Return average sentiment score
        if not scores:
            return 0.0
            
        return sum(scores) / len(scores)

# Helper function
_analyzer = None

def get_sentiment(headlines: List[str]) -> float:
    """Convenience function to get sentiment, managing the analyzer instance."""
    global _analyzer
    if not headlines:
        return 0.0
    if _analyzer is None:
        _analyzer = FinBERTSentiment()
    return _analyzer.score_headlines(headlines)
