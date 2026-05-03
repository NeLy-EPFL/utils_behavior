"""Ball-pushing experiment utilities.

Tools to load, process, and analyze ball-pushing experiments (Drosophila
behavior). Requires the ``video`` and ``plots`` optional extras for full
functionality.
"""

# Submodules are not auto-imported because they pull in moviepy, pygame,
# cv2, etc. Import them explicitly:
#     from utils_behavior.ballpushing import core, dataset, transform_contacts
__all__: list[str] = []
