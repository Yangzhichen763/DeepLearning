import matplotlib.pyplot as plt


def savefig(*, fig: plt.Figure, path: str, file_format: str = "svg", dpi: int = 150):
    path = f"{path}.{file_format}"
    print(f"Saving figure to {path}")
    fig.savefig(path, format=file_format, dpi=dpi)
