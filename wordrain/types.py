from typing import NamedTuple

class WordScore(NamedTuple):
    score: float
    word: str
    force_inclusion: bool
    freq: float
    relfreq: float

class WordInfo(NamedTuple):
    score: float
    word: str
    x: float
    freq: float
    relfreq: float
