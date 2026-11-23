"""Inspect saved voxel-flow evaluation volumes, reports, and diagnostics."""

import argparse
import base64
import gzip
import json
import os
import secrets
import sys
import threading
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml

# ROOT is phase_1/: code lives in ROOT/code, runs and checkpoints in ROOT.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from scripts.voxel_flow import load_model
from src.report_encoder import load_biomedclip_text
from src.voxel_flow.config import load_config
from src.voxel_flow.data import CONTRASTS
from src.voxel_flow.data import load_voxel_target
from src.voxel_flow.data import validate_affine
from src.voxel_flow.flow import predict_clean
from src.voxel_flow.flow import generate_from_noise
from util.report_cache import ReportFeatureCache

from util.biomedclip_tokens import extract_report_tokens

DEFAULT_EVALUATION = ROOT / "runs/voxel_flow_v2_3_5000_seed42/evaluation_epoch_200"


@st.cache_data(max_entries=8)
def load_volume(path, modified_ns):
    """Load a canonical RAS volume and its physical voxel spacing."""

    image = nib.as_closest_canonical(nib.load(path))
    volume = image.get_fdata(dtype=np.float32)

    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f"Expected a finite 3D volume: {path}")

    return volume, image.affine, image.header.get_zooms()[:3]


@st.cache_data(max_entries=4)
def real_comparison_volume(cache_path, modified_ns):
    """Use the cached training units, not the differently normalized source MRI."""

    volume, affine, _ = load_voxel_target(cache_path, (240, 240, 168))
    image = nib.Nifti1Image(volume, affine)

    image.header.set_xyzt_units("mm")

    return gzip.compress(image.to_bytes()), affine


@st.cache_resource(max_entries=1)
def load_generator(checkpoint_path, weights):
    config = load_config(Path(checkpoint_path).parent / "config.yaml")
    config["device"] = "cuda:0"
    model = load_model(config, checkpoint_path, weights)

    model.requires_grad_(False)

    features = ReportFeatureCache(config["report_token_cache"], memory_map=True)

    return model, config, features, threading.Lock()


@st.cache_data
def validation_reports(manifest):
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    frame = frame.loc[frame["split"] == "validation"].reset_index(drop=True)

    if frame.empty:
        raise ValueError("The checkpoint manifest has no validation reports")

    return frame


@st.cache_resource(max_entries=1)
def load_custom_encoder():
    config = yaml.safe_load((ROOT / "code/configs/report_encoder.yaml").read_text())

    for key in ("biomedclip_directory", "biomedbert_directory"):
        config[key] = str(ROOT / config[key])

    # Keep the frozen text tower on CPU to leave GPU memory for MRI synthesis.
    config["device"] = "cpu"
    model, tokenizer = load_biomedclip_text(config)

    return model, tokenizer, config["biomedclip_revision"], threading.Lock()


def display_window(volume):
    """Estimate a display window from positive foreground, excluding background."""

    positive = volume[volume > 0]

    if not len(positive):
        return (0.0, 1.0)

    cutoff = 0.05 * float(np.percentile(positive, 99))
    foreground = positive[positive > cutoff]
    lower, upper = np.percentile(foreground, [2, 98])

    if upper <= lower:
        return (0.0, max(float(upper), 1.0))

    return (float(lower), float(upper))


def generation_panel(directory, checkpoint):
    config = load_config(checkpoint.parent / "config.yaml")
    reports = validation_reports(str(checkpoint.parent / "manifest.csv"))
    controls, output = st.columns([1, 2], gap="large")

    with controls:
        st.subheader("1 · Choose a report")

        source = st.radio(
            "Report source", ["Validation library", "Custom report"], horizontal=True
        )

        search = st.text_input(
            "Search validation scans", placeholder="Study, patient, or words in report"
        )

        matches = reports

        if search.strip():
            mask = (
                reports[["study_uid", "patient_uid", "report"]]
                .apply(
                    lambda column: column.str.contains(
                        search.strip(), case=False, regex=False
                    )
                )
                .any(axis=1)
            )

            matches = reports.loc[mask]

        st.caption(f"{len(matches)} of {len(reports)} validation scans")

        report_index = st.selectbox(
            "Validation report"
            if source == "Validation library"
            else "Reference geometry",
            matches.index.tolist(),
            format_func=lambda i: (
                f"{reports.loc[i, 'study_uid']} · {reports.loc[i, 'series_id']} "
                f"· {reports.loc[i, 'contrast']}"
            ),
            help="For custom text, this scan supplies geometry only.",
        )

        if report_index is None:
            st.session_state.pop("conditioning_report", None)
            st.info("No scans match. Clear the search to continue.")

            return

        reference = reports.loc[report_index]

        if source == "Custom report":
            if st.button("Copy selected validation report into editor"):
                st.session_state.custom_report = reference.report

            report_text = st.text_area(
                "Write or paste your report",
                key="custom_report",
                height=260,
                placeholder="Clinical information, findings, and impression…",
            )

            st.caption(
                "Custom text is encoded locally with the frozen training encoder."
            )
        else:
            report_text = reference.report

            st.caption("Read the full selected text in the Reports tab.")

        st.session_state.conditioning_report = {
            "source": source,
            "report": report_text,
            "reference_study_uid": reference.study_uid,
            "patient_uid": reference.patient_uid,
            "series_id": reference.series_id,
        }

        st.subheader("2 · Generate")

        contrast = st.selectbox(
            "MRI contrast",
            CONTRASTS,
            index=CONTRASTS.index(reference.contrast),
            key=f"generation_contrast_{reference.study_uid}",
        )

        left, right = st.columns([2, 1])

        if "noise_seed" not in st.session_state:
            st.session_state.noise_seed = 42

        if right.button("New seed", use_container_width=True):
            st.session_state.noise_seed = secrets.randbelow(2**31)

        seed = left.number_input("Noise seed", 0, 2**31 - 1, key="noise_seed")
        steps = st.slider("Sampling steps", 1, 100, 16)

        with st.expander("Advanced generation settings"):
            guidance = st.slider("Report guidance scale", 0.0, 5.0, 1.0, 0.1)
            weights = st.selectbox("Checkpoint weights", ["ema", "model"])

            st.caption(f"{checkpoint.name} · Heun sampler, final Euler step")

        generate = st.button(
            "Generate MRI",
            type="primary",
            use_container_width=True,
            disabled=not report_text.strip(),
        )

    with output:
        st.subheader("3 · Explore the MRI")

        if generate:
            try:
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is unavailable")

                with st.spinner("Preparing report and generating MRI on the GPU…"):
                    model, config, features, lock = load_generator(
                        str(checkpoint), weights
                    )

                    config = dict(config, guidance_scale=float(guidance))

                    geometry = nib.load(
                        Path(config["dataset_root"]) / reference.image_path
                    )

                    validate_affine(geometry.affine)

                    if geometry.shape != tuple(config["native_shape"]):
                        raise ValueError(
                            "Reference geometry differs from the model grid"
                        )

                    encoder_revision = None

                    if source == "Custom report":
                        encoder, tokenizer, encoder_revision, encoder_lock = (
                            load_custom_encoder()
                        )

                        with encoder_lock:
                            encoded = extract_report_tokens(
                                encoder, tokenizer, report_text, "cpu"
                            )

                        tokens = encoded["tokens"][encoded["attention_mask"]]
                    else:
                        index = features.index[reference.study_uid]

                        tokens = features.tokens[
                            int(features.offsets[index]) : int(
                                features.offsets[index + 1]
                            )
                        ]

                    if (
                        tokens.ndim != 2
                        or tokens.shape[1] != 768
                        or not torch.isfinite(tokens).all()
                    ):
                        raise ValueError("Invalid report token features")

                    condition = {
                        "report_tokens": tokens[None].to("cuda:0"),
                        "report_mask": torch.ones(
                            1, len(tokens), dtype=torch.bool, device="cuda:0"
                        ),
                        "contrast_id": torch.tensor(
                            [CONTRASTS.index(contrast)], device="cuda:0"
                        ),
                        "report_drop": torch.zeros(
                            1, dtype=torch.bool, device="cuda:0"
                        ),
                    }

                    affine = torch.from_numpy(geometry.affine)[None].to("cuda:0")
                    progress = st.progress(0.0, text="Sampling from random noise…")
                    started = perf_counter()

                    with lock, torch.inference_mode():
                        generator = torch.Generator(device="cuda:0").manual_seed(
                            int(seed)
                        )

                        noise = (
                            torch.randn(
                                1,
                                1,
                                *config["native_shape"],
                                device="cuda:0",
                                generator=generator,
                            )
                            * config["source_noise_std"]
                        )

                        generated, calls = generate_from_noise(
                            lambda state, times: predict_clean(
                                model, state, times, condition, affine, config
                            ),
                            noise,
                            int(steps),
                            observer=lambda time, *_: progress.progress(
                                min(time + 1 / steps, 1.0)
                            ),
                        )

                        array = generated[0, 0].cpu().numpy()

                    if not np.isfinite(array).all():
                        raise ValueError("Generated volume contains nonfinite values")

                    image = nib.Nifti1Image(array, geometry.affine)

                    image.header.set_xyzt_units("mm")

                    lower, upper = display_window(array)
                    image.header["cal_min"] = lower
                    image.header["cal_max"] = upper

                    metadata = {
                        "checkpoint": str(checkpoint),
                        "weights": weights,
                        "seed": int(seed),
                        "steps": int(steps),
                        "solver": "heun_last_euler",
                        "guidance_scale": float(guidance),
                        "report_source": source,
                        "report": report_text,
                        "study_uid": reference.study_uid
                        if source == "Validation library"
                        else None,
                        "geometry_study_uid": reference.study_uid,
                        "geometry_series_id": reference.series_id,
                        "contrast": contrast,
                        "affine": geometry.affine.tolist(),
                        "model_evaluations": calls,
                        "source_noise_std": config["source_noise_std"],
                        "report_token_cache": config["report_token_cache"],
                        "custom_encoder_revision": encoder_revision,
                        "intensity_units": "positive-brain-median normalized",
                        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "mixed_precision": config["mixed_precision"],
                        "reproducibility": (
                            "Seeded input noise; GPU operations may be nondeterministic"
                        ),
                        "seconds": perf_counter() - started,
                    }

                    st.session_state.live_result = {
                        "nifti": gzip.compress(image.to_bytes()),
                        "metadata": metadata,
                    }

                    progress.empty()
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                st.error(str(error))

        if "live_result" not in st.session_state:
            st.info(
                (
                    "Select a validation report or paste custom text, then choose "
                    "Generate MRI."
                )
            )

            st.caption(
                "240 × 240 × 168 native voxels · Positive-brain-median intensity units"
            )

            return

        result = st.session_state.live_result
        metadata = result["metadata"]
        metrics = st.columns(3)

        metrics[0].metric("Seed", metadata["seed"])

        metrics[1].metric("Steps", metadata["steps"])

        metrics[2].metric("MRI contrast", metadata["contrast"])

        st.caption(
            "Displayed MRI uses the report recorded in Reports → Displayed MRI report."
        )

        template = Path(__file__).with_name("brain_viewer.html").read_text()
        encoded = base64.b64encode(result["nifti"]).decode("ascii")
        template = template.replace("__VOLUME_BASE64__", encoded)

        if metadata.get("study_uid") is not None:
            view = st.radio(
                "MRI view",
                ["Generated MRI", "Real + generated"],
                horizontal=True,
            )

            if view == "Real + generated":
                # Resolve the displayed sample, even if selection controls changed.
                matches = reports.loc[
                    (reports.study_uid == metadata["geometry_study_uid"])
                    & (reports.series_id == metadata["geometry_series_id"])
                ]

                if len(matches) != 1:
                    st.error(
                        "Cannot uniquely match the displayed MRI to a validation scan"
                    )

                    return

                real_row = matches.iloc[0]

                cache_path = (
                    Path(config["voxel_cache_root"]) / real_row.voxel_cache_path
                )

                real_bytes, real_affine = real_comparison_volume(
                    str(cache_path),
                    cache_path.stat().st_mtime_ns,
                )

                if not np.allclose(real_affine, metadata["affine"]):
                    st.error("Real and generated MRI grids differ")

                    return

                template = template.replace(
                    "__REFERENCE_BASE64__",
                    base64.b64encode(real_bytes).decode("ascii"),
                )

                st.caption(
                    f"Validation study {real_row.study_uid} · "
                    "Same physical grid and intensity window. "
                    "Generated anatomy is a conditional sample, not a reconstruction."
                )

                if real_row.contrast != metadata["contrast"]:
                    st.info(
                        f"Real MRI contrast: {real_row.contrast}; "
                        f"generated contrast: {metadata['contrast']}."
                    )

        st.iframe(template, height=730)

        st.caption(
            "Open Display settings for window and gamma controls. "
            "Use Full screen to enlarge the MRI viewer (Esc to exit). "
            "Click slices for crosshairs; scroll for slices; drag 3D to rotate. "
            "Display adjustments do not change exported voxel values."
        )

        name = f"voxel_seed{metadata['seed']}_steps{metadata['steps']}"
        downloads = st.columns(2)

        downloads[0].download_button(
            "Download MRI",
            result["nifti"],
            f"{name}.nii.gz",
            "application/gzip",
            use_container_width=True,
        )

        downloads[1].download_button(
            "Download settings + report",
            json.dumps(metadata, indent=2),
            f"{name}.json",
            "application/json",
            use_container_width=True,
        )

        st.caption(
            "The seed fixes input noise; GPU/bfloat16 operations can vary between runs."
        )


def reports_panel():
    current = st.session_state.get("conditioning_report")

    st.subheader("Selected report")

    if current:
        st.caption(
            f"{current['source']} · "
            f"Reference study {current['reference_study_uid']} · "
            f"{current['series_id']}"
        )

        st.text_area(
            "Selected report text", current["report"], height=400, disabled=True
        )

    st.divider()
    st.subheader("Displayed MRI report")

    if "live_result" in st.session_state:
        metadata = st.session_state.live_result["metadata"]

        st.caption(
            f"Seed {metadata['seed']} · {metadata['contrast']} · "
            f"{metadata.get('report_source', 'Validation library')}"
        )

        st.text_area(
            "Exact report used for this MRI",
            metadata["report"],
            height=400,
            disabled=True,
        )
    else:
        st.info("Generate an MRI to record its conditioning report here.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Default: training/epoch_200.pt next to the evaluation directory",
    )

    args = parser.parse_args()
    directory = args.evaluation_dir.resolve()

    checkpoint = (
        args.checkpoint or directory.parent / "training" / "epoch_200.pt"
    ).resolve()

    st.set_page_config(page_title="Voxel-flow evaluation", layout="wide")
    st.title("Brain synthesis studio")
    st.caption(str(directory))

    st.sidebar.caption(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}. "
        "Live generation uses logical cuda:0 on the visible GPU."
    )

    if not directory.is_dir():
        st.error(f"Evaluation directory does not exist: {directory}")
        st.stop()

    settings = json.loads((directory / "settings.json").read_text())
    selected = pd.read_csv(directory / "selected_cases.csv", dtype=str).fillna("")
    real_paths = sorted(directory.glob("case_*_real.nii.gz"))

    if not real_paths:
        st.error("No saved real volumes were found.")
        st.stop()

    cases = [path.name.removesuffix("_real.nii.gz") for path in real_paths]

    live, reports, viewer, diagnostics, configuration = st.tabs(
        ["Generate & explore", "Reports", "Saved evaluation", "Metrics", "Settings"]
    )

    with live:
        generation_panel(directory, checkpoint)

    with reports:
        reports_panel()

    with viewer:
        case = st.selectbox("Saved evaluation case", cases)
        row = selected.iloc[int(case.split("_")[1])]

        with st.expander("Saved case report"):
            st.text(row.report)

        paths = [directory / f"{case}_real.nii.gz"]
        paths += sorted(directory.glob(f"{case}_generated.nii.gz"))
        paths += sorted(directory.glob(f"{case}_seed_*.nii.gz"))
        volumes = [load_volume(str(path), path.stat().st_mtime_ns) for path in paths]
        real, affine, spacing = volumes[0]

        for volume, other_affine, _ in volumes[1:]:
            if volume.shape != real.shape or not np.allclose(affine, other_affine):
                st.error(
                    "Saved volumes have different physical grids; comparison stopped."
                )

                st.stop()

        if len(paths) == 1:
            st.info("This reference case has no saved generated volume.")

        st.caption(
            f"Canonical RAS grid: {real.shape}; spacing (mm): "
            f"{tuple(round(float(value), 3) for value in spacing)}. "
            "Slices use neurological display: image right is patient right. "
            "Generated and real images are conditional samples, not paired "
            "reconstructions."
        )

        limits = display_window(real)
        window_controls = st.columns(2)

        lower = window_controls[0].number_input(
            "Display minimum", value=float(limits[0])
        )

        upper = window_controls[1].number_input(
            "Display maximum", value=float(limits[1])
        )

        if lower >= upper:
            st.error("Display maximum must exceed display minimum.")
            st.stop()

        indices = []

        for axis, name in enumerate(
            ["Sagittal (L to R)", "Coronal (P to A)", "Axial (I to S)"]
        ):
            indices.append(
                st.slider(name, 0, real.shape[axis] - 1, real.shape[axis] // 2)
            )

        # A shared intensity window preserves differences between saved samples.
        fig, axes = plt.subplots(
            3, len(paths), squeeze=False, figsize=(5 * len(paths), 12)
        )

        planes = [("Sagittal", 1, 2), ("Coronal", 0, 2), ("Axial", 0, 1)]

        for column, (path, (volume, _, _)) in enumerate(zip(paths, volumes)):
            for axis, (plane, horizontal, vertical) in enumerate(planes):
                panel = axes[axis, column]

                panel.imshow(
                    np.take(volume, indices[axis], axis=axis).T,
                    origin="lower",
                    cmap="gray",
                    vmin=lower,
                    vmax=upper,
                    aspect=float(spacing[vertical] / spacing[horizontal]),
                    interpolation="nearest",
                )

                panel.set_title(
                    f"{path.name.removesuffix('.nii.gz')}\n{plane} {indices[axis]}"
                )

                panel.set_xlabel(["P → A", "L → R", "L → R"][axis])
                panel.set_ylabel(["I → S", "I → S", "P → A"][axis])

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with diagnostics:
        st.caption(
            "Metrics below are the saved evaluation results; they are not recomputed."
        )

        for name in [
            "per_case_scalars",
            "prediction_diagnostics",
            "trajectory_diagnostics",
            "anatomy_report_review",
        ]:
            with st.expander(
                name.replace("_", " ").title(), expanded=name == "per_case_scalars"
            ):
                frame = pd.read_csv(directory / f"{name}.csv")

                if "case" in frame.columns:
                    only_case = st.checkbox(
                        "Show selected case only", value=True, key=name
                    )

                    if only_case:
                        frame = frame.loc[frame["case"] == case]

                st.dataframe(frame, hide_index=True, width="stretch")

        with st.expander("Full aggregate metrics"):
            st.json(json.loads((directory / "metrics.json").read_text()))

        with st.expander("Saved evaluation preview"):
            st.image(str(directory / "preview.png"))

    with configuration:
        st.json(settings)
        st.dataframe(selected, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
