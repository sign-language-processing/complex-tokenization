from graphviz import Source

from PIL import Image
from io import BytesIO


def draw_dot_content(dot_content: str) -> Image:
    dot = """
digraph G {
  graph [compound=true, rankdir=LR, fontsize=16, nodesep=0.6];
  node  [shape=circle, fontsize=16];
  edge  [fontsize=12, arrowhead=none]; // default: no arrowheads
""" + dot_content + "\n}"
    src = Source(dot)

    png_bytes = src.pipe(format="png")

    return Image.open(BytesIO(png_bytes))


def create_gif(frames: list[Image.Image], save=None) -> Image.Image:
    target = save if save is not None else BytesIO()
    frames[0].save(
        target,
        format="GIF",
        save_all=True,
        append_images=frames[1:],  # skip the first one (it's already saved)
        duration=500,
        loop=0,
        disposal=2,                # <-- clear previous frame
    )
    if isinstance(target, BytesIO):
        target.seek(0)
    return Image.open(target)