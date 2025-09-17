"""Dataset downloading and preprocessing utilities for VLA project.

This module provides functionality to download and preprocess various vision-language datasets
including LLaVA datasets, COCO, GQA, OCR-VQA, TextVQA, and Visual Genome datasets. It handles
automatic downloading with progress bars, extraction of compressed archives, and image format
conversion.

Example:
    Basic usage for downloading dataset components:

    ```python
    from vla_project.preprocessing.download import download_with_progress, extract_with_progress

    # Download a file
    file_path = download_with_progress("https://example.com/data.zip", Path("./downloads"))

    # Extract an archive
    extract_path = extract_with_progress(file_path, Path("./data"), "directory")
    ```
"""

from pathlib import Path
from typing import TypedDict
from zipfile import ZipFile

import requests
from PIL import Image
from rich.progress import BarColumn, DownloadColumn, MofNCompleteColumn, Progress, TextColumn, TransferSpeedColumn
from tqdm import tqdm

from vla_project.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


class DatasetComponent(TypedDict, total=False):
    """Type definition for dataset component configuration.

    Attributes:
        name: The name or relative path where the component should be stored.
        extract: Whether the downloaded file needs to be extracted.
        extract_type: Type of extraction ("file" or "directory").
        url: The URL to download the component from.
        do_rename: Whether to rename the downloaded/extracted content.
    """

    name: str
    extract: bool
    extract_type: str
    url: str
    do_rename: bool


DATASET_COMPONENTS: dict[str, list[DatasetComponent]] = {
    """Dictionary containing dataset component configurations for different datasets.
    
    This dictionary defines the components (files and directories) that need to be downloaded
    for various vision-language datasets. Each dataset has a list of components with their
    download URLs, extraction requirements, and storage configurations.
    
    Supported datasets:
        - llava-laion-cc-sbu-558k: LLaVA pretraining dataset with chat traces and images
        - llava-v1.5-instruct: LLaVA v1.5 instruction following dataset with multiple image sources
    
    Keys:
        Dataset identifier string
    
    Values:
        List of DatasetComponent configurations for each component
    """
    # === LLaVa v1.5 Dataset(s) ===
    # Note =>> This is the full suite of datasets included in the LLaVa 1.5 "finetuning" stage; all the LLaVa v1.5
    #          models are finetuned on this split. We use this dataset for all experiments in our paper.
    "llava-laion-cc-sbu-558k": [
        {
            "name": "chat.json",  # Contains the "chat" traces :: {"human" => <prompt>, "gpt" => <caption>}
            "extract": False,
            "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json",
            "do_rename": True,
        },
        {
            "name": "images",  # Contains the LLaVa Processed Images (jpgs, 224x224 resolution)
            "extract": True,
            "extract_type": "directory",
            "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip",
            "do_rename": False,
        },
    ],
    "llava-v1.5-instruct": [
        {
            "name": "llava_v1_5_mix665k.json",
            "extract": False,
            "url": (
                "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json"
            ),
            "do_rename": True,
        },
        {
            "name": "coco/train2017",  # Visual Instruct Tuning images are all sourced from COCO Train 2017
            "extract": True,
            "extract_type": "directory",
            "url": "http://images.cocodataset.org/zips/train2017.zip",
            "do_rename": True,
        },
        {
            "name": "gqa/images",
            "extract": True,
            "extract_type": "directory",
            "url": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
            "do_rename": True,
        },
        {
            "name": "ocr_vqa/images",
            "extract": True,
            "extract_type": "directory",
            "url": "https://huggingface.co/datasets/qnguyen3/ocr_vqa/resolve/main/ocr_vqa.zip",
            "do_rename": True,
        },
        {
            "name": "textvqa/train_images",
            "extract": True,
            "extract_type": "directory",
            "url": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
            "do_rename": True,
        },
        {
            "name": "vg/VG_100K",
            "extract": True,
            "extract_type": "directory",
            "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
            "do_rename": True,
        },
        {
            "name": "vg/VG_100K_2",
            "extract": True,
            "extract_type": "directory",
            "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
            "do_rename": True,
        },
    ],
}


def convert_to_jpg(image_dir: Path) -> None:
    """Convert all images in a directory to JPG format.

    Specifically handles OCR-VQA Images by iterating through a directory and converting
    all GIF and PNG images to JPG format. This is necessary for dataset preprocessing
    as some datasets contain mixed image formats.

    Args:
        image_dir: Path to the directory containing images to convert.

    Raises:
        ValueError: If an unsupported image format is encountered.

    Note:
        - Only converts GIF and PNG files to JPG
        - Skips files that are already in JPG/JPEG format
        - Skips files where a JPG version already exists
        - For GIF files, only the first frame is converted
    """
    overwatch.info(f"Converting all images in {image_dir} to JPG format...")
    for image_fn in tqdm(list(image_dir.iterdir())):
        if image_fn.suffix in {".jpg", "jpeg"} or (jpg_fn := image_dir / f"{image_fn.stem}.jpg").exists():
            continue
        if image_fn.suffix == ".gif":
            gif = Image.open(image_fn)
            gif.seek(0)
            gif.convert("RGB").save(jpg_fn)
        elif image_fn.suffix in {".png", ".PNG"}:
            Image.open(image_fn).convert("RGB").save(jpg_fn)
        else:
            raise ValueError(f"Unsupported image format: {image_fn.suffix}")


def download_with_progress(url: str, download_dir: Path, chunk_size_bytes: int = 1024) -> Path:
    """Download files from the internet with a Rich-based progress bar.

    Downloads a file from the given URL to the specified directory, displaying
    a progress bar with download speed, percentage completion, and transfer size.
    If the destination file already exists, the download is skipped.

    Args:
        url: The URL to download the file from.
        download_dir: The directory where the downloaded file should be saved.
        chunk_size_bytes: The size of each chunk to download in bytes. Defaults to 1024.

    Returns:
        Path to the downloaded file.

    Raises:
        requests.RequestException: If the HTTP request fails.
        IOError: If there's an error writing the file to disk.

    Example:
        >>> download_dir = Path("./downloads")
        >>> download_dir.mkdir(exist_ok=True)
        >>> file_path = download_with_progress(
        ...     "https://example.com/data.zip",
        ...     download_dir
        ... )
        >>> print(f"Downloaded to: {file_path}")
    """
    overwatch.info(f"Downloading {(dest_path := download_dir / Path(url).name)} from `{url}`", ctx_level=1)
    if dest_path.exists():
        return dest_path

    # Otherwise --> fire an HTTP Request, with `stream = True` (to avoid loading all in memory)
    response = requests.get(url, stream=True)

    # Download w/ Transfer-Aware Progress
    #   => Reference: https://github.com/Textualize/rich/blob/master/examples/downloader.py
    with Progress(
        TextColumn("[bold]{task.description} - {task.fields[fname]}"),  # display task description and filename
        BarColumn(bar_width=None),  # display progress bar (auto width)
        "[progress.percentage]{task.percentage:>3.1f}%",  # display percentage completion (right-aligned)
        "•",  # display a separator
        DownloadColumn(),  # display downloaded / total size
        "•",  # display a separator
        TransferSpeedColumn(),  # display current transfer speed
        transient=True,  # hide when done
    ) as dl_progress:
        # add a new task to the progress bar
        dl_tid = dl_progress.add_task(
            "Downloading", fname=dest_path.name, total=int(response.headers.get("content-length", "None"))
        )
        with open(dest_path, "wb") as f:
            for data in response.iter_content(chunk_size=chunk_size_bytes):
                # update task (step by size of data read)
                dl_progress.advance(dl_tid, f.write(data))

    return dest_path


def extract_with_progress(archive_path: Path, download_dir: Path, extract_type: str, cleanup: bool = False) -> Path:
    """Extract compressed archives with a Rich-based progress bar.

    Extracts ZIP archives to the specified directory, displaying a progress bar
    with extraction progress. Supports both single file and directory extraction.

    Args:
        archive_path: Path to the ZIP archive to extract.
        download_dir: Directory where the archive contents should be extracted.
        extract_type: Type of extraction, either "file" for single file or "directory" for multiple files.
        cleanup: Whether to delete the archive file after extraction. Defaults to False.

    Returns:
        Path to the extracted content (first extracted file/directory).

    Raises:
        AssertionError: If the archive is not a ZIP file or if extract_type is "file" but
                       archive contains multiple files.
        ValueError: If extract_type is not "file" or "directory".
        zipfile.BadZipFile: If the archive is corrupted or invalid.

    Example:
        >>> archive_path = Path("./data.zip")
        >>> extract_dir = Path("./extracted")
        >>> extract_dir.mkdir(exist_ok=True)
        >>> extracted_path = extract_with_progress(
        ...     archive_path,
        ...     extract_dir,
        ...     "directory",
        ...     cleanup=True
        ... )
        >>> print(f"Extracted to: {extracted_path}")
    """
    assert archive_path.suffix == ".zip", "Only `.zip` compressed archives are supported for now!"
    overwatch.info(f"Extracting {archive_path.name} to `{download_dir}`", ctx_level=1)

    # Extract w/ Progress
    with Progress(
        TextColumn("[bold]{task.description} - {task.fields[aname]}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        transient=True,
    ) as ext_progress:
        with ZipFile(archive_path) as zf:
            ext_tid = ext_progress.add_task("Extracting", aname=archive_path.name, total=len(members := zf.infolist()))
            extract_path = Path(zf.extract(members[0], download_dir))
            if extract_type == "file":
                assert len(members) == 1, f"Archive `{archive_path}` with extract type `{extract_type} has > 1 member!"
            elif extract_type == "directory":
                for member in members[1:]:
                    zf.extract(member, download_dir)
                    ext_progress.advance(ext_tid)
            else:
                raise ValueError(f"Extract type `{extract_type}` for archive `{archive_path}` is not defined!")

    # Cleanup (if specified)
    if cleanup:
        archive_path.unlink()

    return extract_path
