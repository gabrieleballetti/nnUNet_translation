from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SiteConfig:
    key: str
    source_id: int
    source_name: str
    target_id: int
    target_name: str

    @property
    def source_dataset(self) -> str:
        return f"Dataset{self.source_id:03d}_{self.source_name}"

    @property
    def target_dataset(self) -> str:
        return f"Dataset{self.target_id:03d}_{self.target_name}"


SITES = {
    "AB": SiteConfig("AB", 101, "SynthRAD2025_AB_MR", 102, "SynthRAD2025_AB_CT"),
    "TH": SiteConfig("TH", 105, "SynthRAD2025_TH_MR", 106, "SynthRAD2025_TH_CT"),
}

DEFAULT_PATCH_SIZE = (48, 192, 224)
DEFAULT_PLANS_NAME = "nnResUNetPlans_ABTH_48x192x224_CTNorm"


def as_env_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def nnunet_executable(name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    candidate = Path(sys.executable).with_name(name + suffix)
    return str(candidate) if candidate.exists() else name


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def dataset_json(channel_name: str, num_training: int) -> dict:
    return {
        "channel_names": {"0": channel_name},
        "labels": {"background": 0, "label_001": 1},
        "numTraining": num_training,
        "file_ending": ".mha",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def write_json(path: Path, payload: object, dry_run: bool) -> None:
    print(f"write {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def place_file(src: Path, dst: Path, mode: str, overwrite: bool, dry_run: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        if not dry_run:
            dst.unlink()
    print(f"{mode} {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def iter_cases(data_root: Path, site: SiteConfig) -> list[Path]:
    site_root = data_root / site.key
    if not site_root.is_dir():
        raise FileNotFoundError(f"Missing synthRAD site folder: {site_root}")
    cases = sorted(
        case for case in site_root.iterdir()
        if case.is_dir()
        and (case / "mr.mha").is_file()
        and (case / "ct.mha").is_file()
        and (case / "mask.mha").is_file()
    )
    if not cases:
        raise RuntimeError(f"No complete mr/ct/mask cases found in {site_root}")
    return cases


def make_split(case_ids: list[str], seed: int) -> list[dict[str, list[str]]]:
    shuffled = case_ids[:]
    random.Random(seed).shuffle(shuffled)
    num_train = int(len(shuffled) * 0.8)
    return [{"train": sorted(shuffled[:num_train]), "val": sorted(shuffled[num_train:])}]


def prepare_raw(args: argparse.Namespace, raw_root: Path) -> None:
    for site_key in args.sites:
        site = SITES[site_key]
        cases = iter_cases(args.data_root, site)
        source_root = raw_root / site.source_dataset
        target_root = raw_root / site.target_dataset

        print(f"Preparing {site.key}: {len(cases)} cases")
        for folder in (source_root / "imagesTr", source_root / "labelsTr", target_root / "imagesTr", target_root / "labelsTr"):
            if not args.dry_run:
                folder.mkdir(parents=True, exist_ok=True)

        for case in cases:
            case_id = case.name
            place_file(case / "mr.mha", source_root / "imagesTr" / f"{case_id}_0000.mha", args.link_mode, args.overwrite, args.dry_run)
            place_file(case / "mask.mha", source_root / "labelsTr" / f"{case_id}.mha", args.link_mode, args.overwrite, args.dry_run)
            place_file(case / "ct.mha", target_root / "imagesTr" / f"{case_id}_0000.mha", args.link_mode, args.overwrite, args.dry_run)
            place_file(case / "mask.mha", target_root / "labelsTr" / f"{case_id}.mha", args.link_mode, args.overwrite, args.dry_run)

        case_ids = [case.name for case in cases]
        split = make_split(case_ids, args.seed)
        write_json(source_root / "dataset.json", dataset_json("MR", len(cases)), args.dry_run)
        write_json(target_root / "dataset.json", dataset_json("CT", len(cases)), args.dry_run)
        write_json(source_root / "splits_final.json", split, args.dry_run)
        write_json(target_root / "splits_final.json", split, args.dry_run)
        print(f"{site.key} split: {len(split[0]['train'])} train / {len(split[0]['val'])} val")


def dataset_ids(sites: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for site_key in sites:
        site = SITES[site_key]
        ids.extend([site.source_id, site.target_id])
    return ids


def plan_without_preprocessing(args: argparse.Namespace, env: dict[str, str]) -> None:
    command = [
        nnunet_executable("nnUNetv2_plan_and_preprocess"),
        "-d",
        *[str(dataset_id) for dataset_id in dataset_ids(args.sites)],
        "-c",
        "3d_fullres",
        "-pl",
        args.planner,
        "-gpu_memory_target",
        str(args.gpu_memory_target),
        "-overwrite_plans_name",
        args.plans_name,
        "--no_pp",
        "--verify_dataset_integrity",
        "-npfp",
        str(args.fingerprint_processes),
    ]
    if args.target_spacing is not None:
        command.extend(["-overwrite_target_spacing", *[str(value) for value in args.target_spacing]])
    run_command(command, env, args.dry_run)


def patch_plans(args: argparse.Namespace, preprocessed_root: Path) -> None:
    patch_size = [int(value) for value in args.patch_size]
    from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

    for site_key in args.sites:
        site = SITES[site_key]
        for dataset_name in (site.source_dataset, site.target_dataset):
            plans_path = preprocessed_root / dataset_name / f"{args.plans_name}.json"
            print(f"patch {plans_path}: 3d_fullres.patch_size = {patch_size}")
            if args.dry_run:
                continue
            with plans_path.open("r", encoding="utf-8") as handle:
                plans = json.load(handle)
            config = plans["configurations"]["3d_fullres"]
            config["patch_size"] = patch_size
            _, strides, kernels, topology_patch_size, shape_must_be_divisible_by = get_pool_and_conv_props(
                config["spacing"],
                patch_size,
                4,
                999999,
            )
            architecture = config["architecture"]["arch_kwargs"]
            num_stages = len(strides)
            max_features = max(architecture["features_per_stage"])
            base_features = architecture["features_per_stage"][0]
            architecture["n_stages"] = num_stages
            architecture["features_per_stage"] = [
                int(min(max_features, base_features * (2 ** stage))) for stage in range(num_stages)
            ]
            architecture["kernel_sizes"] = [[int(axis) for axis in kernel] for kernel in kernels]
            architecture["strides"] = [[int(axis) for axis in stride] for stride in strides]
            architecture["n_blocks_per_stage"] = [
                int(architecture["n_blocks_per_stage"][min(stage, len(architecture["n_blocks_per_stage"]) - 1)])
                for stage in range(num_stages)
            ]
            architecture["n_conv_per_stage_decoder"] = [
                int(architecture["n_conv_per_stage_decoder"][min(stage, len(architecture["n_conv_per_stage_decoder"]) - 1)])
                for stage in range(num_stages - 1)
            ]
            config["patch_size"] = [int(axis) for axis in topology_patch_size]
            print(f"  topology strides={architecture['strides']} divisible_by={[int(axis) for axis in shape_must_be_divisible_by]}")
            if args.batch_size is not None:
                config["batch_size"] = int(args.batch_size)
            with plans_path.open("w", encoding="utf-8") as handle:
                json.dump(plans, handle, indent=2)
                handle.write("\n")


def preprocess(args: argparse.Namespace, env: dict[str, str]) -> None:
    command = [
        nnunet_executable("nnUNetv2_preprocess"),
        "-d",
        *[str(dataset_id) for dataset_id in dataset_ids(args.sites)],
        "-plans_name",
        args.plans_name,
        "-c",
        "3d_fullres",
        "-np",
        str(args.preprocess_processes),
    ]
    run_command(command, env, args.dry_run)


def load_preprocessed_data(case_path: Path) -> np.ndarray:
    npy_path = case_path.with_suffix(".npy")
    if npy_path.is_file():
        return np.load(npy_path)
    npz_path = case_path.with_suffix(".npz")
    if npz_path.is_file():
        return np.load(npz_path)["data"]
    raise FileNotFoundError(f"Could not find {npy_path} or {npz_path}")


def install_ct_targets(args: argparse.Namespace, raw_root: Path, preprocessed_root: Path) -> None:
    for site_key in args.sites:
        site = SITES[site_key]
        source_preprocessed = preprocessed_root / site.source_dataset
        target_preprocessed = preprocessed_root / site.target_dataset
        source_data_folder = source_preprocessed / "nnUNetPlans_3d_fullres"
        target_data_folder = target_preprocessed / "nnUNetPlans_3d_fullres"
        source_gt_folder = source_preprocessed / "gt_segmentations"
        target_images = raw_root / site.target_dataset / "imagesTr"

        cases = sorted(path.stem[:-5] for path in target_images.glob("*_0000.mha"))
        print(f"Installing CT targets for {site.key}: {len(cases)} cases")
        if not args.dry_run:
            source_gt_folder.mkdir(parents=True, exist_ok=True)

        for case_id in cases:
            target_case_path = target_data_folder / case_id
            source_seg_path = source_data_folder / f"{case_id}_seg.npy"
            print(f"target array {target_case_path}.npy/.npz -> {source_seg_path}")
            if not args.dry_run:
                target_array = load_preprocessed_data(target_case_path)
                np.save(source_seg_path, target_array)

            source_ct = target_images / f"{case_id}_0000.mha"
            destination_ct = source_gt_folder / f"{case_id}.mha"
            place_file(source_ct, destination_ct, "copy", True, args.dry_run)


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare synthRAD2025 abdomen/thorax nnU-Net translation datasets.")
    parser.add_argument("--data-root", type=Path, default=repo_root / "data" / "synthRAD2025_Task1_Train" / "Task1")
    parser.add_argument("--nnunet-root", type=Path, default=repo_root / "nnunet_data")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--preprocessed-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--sites", nargs="+", choices=sorted(SITES), default=["AB", "TH"])
    parser.add_argument("--patch-size", nargs=3, type=int, default=DEFAULT_PATCH_SIZE, metavar=("Z", "Y", "X"))
    parser.add_argument("--target-spacing", nargs=3, type=float, default=None, metavar=("Z", "Y", "X"))
    parser.add_argument("--plans-name", default=DEFAULT_PLANS_NAME)
    parser.add_argument("--planner", default="nnUNetPlannerResUNet")
    parser.add_argument("--gpu-memory-target", type=float, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20250626)
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--fingerprint-processes", type=int, default=4)
    parser.add_argument("--preprocess-processes", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-raw", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--patch-plans", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--install-targets", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run prepare, plan, patch, preprocess, and install targets.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    raw_root = args.raw_root or args.nnunet_root / "raw"
    preprocessed_root = args.preprocessed_root or args.nnunet_root / "preprocessed"
    results_root = args.results_root or args.nnunet_root / "results"
    for path in (raw_root, preprocessed_root, results_root):
        if not args.dry_run:
            path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = as_env_path(raw_root)
    env["nnUNet_preprocessed"] = as_env_path(preprocessed_root)
    env["nnUNet_results"] = as_env_path(results_root)

    print("nnUNet_raw=", env["nnUNet_raw"])
    print("nnUNet_preprocessed=", env["nnUNet_preprocessed"])
    print("nnUNet_results=", env["nnUNet_results"])
    print("plans_name=", args.plans_name)
    print("patch_size=", list(args.patch_size))

    no_stage_selected = not any((args.prepare_raw, args.plan, args.patch_plans, args.preprocess, args.install_targets, args.all))
    if args.all or args.prepare_raw or no_stage_selected:
        prepare_raw(args, raw_root)
    if args.all or args.plan:
        plan_without_preprocessing(args, env)
    if args.all or args.patch_plans:
        patch_plans(args, preprocessed_root)
    if args.all or args.preprocess:
        preprocess(args, env)
    if args.all or args.install_targets:
        install_ct_targets(args, raw_root, preprocessed_root)

    print("Done.")


if __name__ == "__main__":
    main()