from pathlib import Path
from typing import List


def rename_pngs_with_prefix(folder: Path, prefix: str, start_index: int = 1) -> None:
    """
    將資料夾內所有 PNG 檔案重新命名為 prefix + num:03d

    Example:
        img_001.png
        img_002.png
        img_003.png
    """

    png_files: List[Path] = sorted(folder.glob("*.png"))

    for i, file_path in enumerate(png_files, start=start_index):
        new_name: str = f"{prefix}_{i:02d}.png"
        new_path: Path = folder / new_name

        print(f"{file_path.name} -> {new_name}")
        file_path.rename(new_path)


if __name__ == "__main__":

    folder_path: Path = Path("SimCases/Urban-1/masks")

    rename_pngs_with_prefix(folder=folder_path, prefix="mask")
