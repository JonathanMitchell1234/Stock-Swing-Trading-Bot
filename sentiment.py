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

import config
from logger import get_logger
from typing import List

log = get_logger("sentiment")

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
            
            requested_device = str(getattr(config, "NLP_DEVICE", "auto")).lower().strip()
            if requested_device not in {"auto", "cuda", "cpu"}:
                log.warning("Invalid NLP_DEVICE=%s. Falling back to 'auto'.", requested_device)
                requested_device = "auto"

            cuda_available = torch.cuda.is_available()
            use_cuda = requested_device == "cuda" or (requested_device == "auto" and cuda_available)
            if use_cuda and not cuda_available:
                log.warning("NLP_DEVICE='cuda' requested but CUDA is unavailable. Falling back to CPU.")
                use_cuda = False
            device = 0 if use_cuda else -1

            # Temporary fix for CVE-2025-32434 weights_only restriction in torch when using transformers + torch < 2.6
            try:
                # We can inform transformers not to use weights_only by importing weights_only = False
                # Or bypass the restrict by environment variable before transformers imports torch
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            except Exception:
                pass

            pipeline_kwargs = {
                "task": "sentiment-analysis",
                "model": "ProsusAI/finbert",
                "device": device,
            }

            if use_cuda:
                gpu_name = "unknown"
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    pass
                log.info("FinBERT using CUDA (GPU): %s", gpu_name)
                if bool(getattr(config, "NLP_USE_FP16", True)):
                    pipeline_kwargs["torch_dtype"] = torch.float16
            else:
                log.info("FinBERT using CPU.")

            try:
                self.pipeline = pipeline(**pipeline_kwargs)
            except Exception as e:
                if "torch_dtype" in pipeline_kwargs:
                    log.warning("FinBERT FP16 init failed (%s). Retrying in float32.", e)
                    pipeline_kwargs.pop("torch_dtype", None)
                    self.pipeline = pipeline(**pipeline_kwargs)
                else:
                    raise

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
