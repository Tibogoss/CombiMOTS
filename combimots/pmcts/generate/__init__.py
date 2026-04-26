"""Generation package exports.

Keep imports lazy so docking utilities can import ``pmcts.generate.node`` for
type annotations without importing the full generation CLI and creating a
docking/generation circular import.
"""

__all__ = ["Generator", "Node", "create_model_scoring_fn", "generate", "save_generated_molecules"]


def __getattr__(name):
    if name == "generate":
        from pmcts.generate.generate import generate
        return generate
    if name == "Generator":
        from pmcts.generate.generator import Generator
        return Generator
    if name == "Node":
        from pmcts.generate.node import Node
        return Node
    if name in {"create_model_scoring_fn", "save_generated_molecules"}:
        from pmcts.generate.utils import create_model_scoring_fn, save_generated_molecules
        return {
            "create_model_scoring_fn": create_model_scoring_fn,
            "save_generated_molecules": save_generated_molecules,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
