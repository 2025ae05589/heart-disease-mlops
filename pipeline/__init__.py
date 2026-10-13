import os

# The steps only save plots to files and never show them on screen. It has to
# be set before Matplotlib is imported.
os.environ.setdefault("MPLBACKEND", "Agg")
