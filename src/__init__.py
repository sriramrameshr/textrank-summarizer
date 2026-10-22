"""Extractive TextRank summarizer package.

Modules
-------
dataset      : download/load the cached CNN/DailyMail slice
preprocess   : sentence splitting, tokenisation, stopword removal, stemming
textrank     : TF-IDF sentence graph + PageRank sentence ranking
baselines    : Lead-k and Random-k baselines
evaluate     : ROUGE-1/2/L scoring and aggregation utilities
visualize    : matplotlib figures used by the notebook, report and app
"""

__version__ = "1.0.0"
