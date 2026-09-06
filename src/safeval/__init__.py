"""safeval: a lightweight, reproducible harness for evaluating LLM safety behavior.

The package measures four dimensions of model safety:

* refusal on mild classic harmful requests,
* helpfulness on benign look-alike requests (over-refusal),
* robustness to simple jailbreak framings, and
* resistance to prompt injection in untrusted text.

Grading is rule-based by default with an optional LLM-as-judge. The harness is
deliberately small and standard-library-first so the mechanics stay legible.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
