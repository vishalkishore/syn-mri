"""Download pathology labels, the MNI template, and the frozen report encoder.

Run after `preprocessing/preprocess.py plan`, which pins the dataset revision.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen
import zipfile

from huggingface_hub import hf_hub_download
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_encoder import download_report_encoder


TEMPLATE_URL = (
    "https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/"
    "mni_icbm152_nlin_asym_09c_nifti.zip"
)


def download_template(template_dir):
    template_dir.mkdir(parents=True, exist_ok=True)

    filenames = [
        "mni_icbm152_t1_tal_nlin_asym_09c.nii",
        "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii",
        "COPYING",
    ]

    if all((template_dir / name).is_file() for name in filenames):
        return

    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "template.zip"

        with (
            urlopen(TEMPLATE_URL, timeout=120) as source,
            archive_path.open("wb") as output,
        ):
            shutil.copyfileobj(source, output)

        with zipfile.ZipFile(archive_path) as archive:
            for name in filenames:
                member = (
                    name if name == "COPYING" else f"mni_icbm152_nlin_asym_09c/{name}"
                )

                destination = template_dir / name

                if destination.exists():
                    continue

                # Copy named members only; do not extract arbitrary archive paths.
                with archive.open(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--root", type=Path, default=Path("data/mr_rate_t1"))
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("data/templates/MNI152NLin2009cAsym"),
    )
    parser.add_argument(
        "--encoder-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/report_encoder.yaml",
    )

    args = parser.parse_args()

    policy = json.loads((args.root / "policy.json").read_text())

    if not policy["revision"]:
        raise ValueError(
            "Plan with a pinned dataset revision before downloading labels"
        )

    # Labels and MRI tables must come from the same planned dataset revision.
    hf_hub_download(
        policy["repo"],
        "pathology_labels/mrrate_labels.csv",
        repo_type="dataset",
        revision=policy["revision"],
        local_dir=args.root / "source",
        token=True,
    )

    download_template(args.template_dir)

    download_report_encoder(yaml.safe_load(args.encoder_config.read_text()))

    print(f"Labels: {args.root / 'source/pathology_labels/mrrate_labels.csv'}")
    print(f"Template: {args.template_dir}")


if __name__ == "__main__":
    main()
