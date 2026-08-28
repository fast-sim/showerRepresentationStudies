import gc
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np


STEP2POINT = Path("/eos/user/s/siyuch/step2point/examples/run_step2point_pipeline.py")
ROOT2H5 = Path("/eos/user/s/siyuch/simulation/step2point_checks/dataset/root2h5.py")
TIME_CUTS = [10.0, 400.0]

# ENERGIES = ["1GeV", "100GeV", "1TeV"]
ENERGIES = ["10GeV", "100GeV"]
# ENERGIES = ["1GeV"]

Y_POS = {
    "ODD": 1250,
    "CLD": 2150,
}

COMPACT_XML = {
    "ODD": f"{os.environ['ODD_INSTALL']}/share/OpenDataDetector/xml/OpenDataDetector.xml",
    "CLD": "/eos/user/s/siyuch/simulation/step2point_checks/simulation/CLD_o2_v07.xml",
}

ALGORITHMS = [
    "identity",
    "merge_within_cell",
    "merge_within_regular_subcell",
    "hdbscan",
]

EXTRA_ARGS = {
    "identity": None,
    "merge_within_cell": None,
    "merge_within_regular_subcell": [
        "--collection-name",
        "ECalBarrelCollection",
        "HCalBarrelCollection",
        "--grid-x",
        "3", "3",
        "--grid-y",
        "3", "3",
        "--position-mode",
        "weighted", "weighted",
    ],
    "hdbscan": [
        "--use-time",
        "--collection-name",
        "ECalBarrelCollection",
        "HCalBarrelCollection",
        "ECalEndcapCollection",
        "HCalEndcapCollection",
    ],
}

EXTRA_ARGS_HADRONIC = {
    "identity": None,
    "merge_within_cell": None,
    "merge_within_regular_subcell": [
        "--collection-name",
        "HCalBarrelCollection",
        "--grid-x",
        "3", 
        "--grid-y",
        "3", 
        "--position-mode",
        "weighted", 
    ],
    "hdbscan": [
        "--use-time",
        "--collection-name",
        "ECalBarrelCollection",
        "HCalBarrelCollection",
        "ECalEndcapCollection",
        "HCalEndcapCollection",
    ],
}


# =========================
# pre-processing
# =========================
def apply_time_cut(
    input_file: str | Path,
    time_cut: float,
) -> Path:
    input_path = Path(input_file)
    output_file = input_path.with_name(
        f"{input_path.stem}_timecut_{time_cut:g}ns"
        f"{input_path.suffix}"
    )

    if time_cut < 0:
        raise ValueError(
            f"time_cut must be non-negative, got {time_cut}"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    with h5py.File(input_path, "r") as source:
        if "steps" not in source:
            raise KeyError(
                f"{input_path} does not contain 'steps'"
            )

        source_steps = source["steps"]

        if "time" not in source_steps:
            raise KeyError(
                f"{input_path} does not contain 'steps/time'"
            )

        if "energy" not in source_steps:
            raise KeyError(
                f"{input_path} does not contain 'steps/energy'"
            )

        times = source_steps["time"][:]
        energy = source_steps["energy"][:]

        if len(times) != len(energy):
            raise ValueError(
                "steps/time and steps/energy have "
                "different lengths: "
                f"{len(times)} != {len(energy)}"
            )

        n_points_before = len(times)

        mask = (
            np.isfinite(times)
            & (times <= time_cut)
        )

        n_points_after = int(
            np.count_nonzero(mask)
        )

        energy_before = float(
            np.sum(energy, dtype=np.float64)
        )
        energy_after = float(
            np.sum(energy[mask], dtype=np.float64)
        )

        # Opening with "w" overwrites an old time-cut file.
        with h5py.File(output_file, "w") as target:
            # Preserve root attributes.
            for key, value in source.attrs.items():
                target.attrs[key] = value

            target.attrs["time_cut_ns"] = time_cut
            target.attrs["source_input"] = str(
                input_path
            )

            # Copy metadata, particles, primary, and any
            # other non-steps groups unchanged.
            for name in source:
                if name != "steps":
                    source.copy(name, target)

            target_steps = target.create_group(
                "steps"
            )

            # Preserve steps-group attributes.
            for key, value in source_steps.attrs.items():
                target_steps.attrs[key] = value

            for name, source_object in source_steps.items():
                # Preserve possible nested groups.
                if not isinstance(
                    source_object,
                    h5py.Dataset,
                ):
                    source_steps.copy(
                        name,
                        target_steps,
                    )
                    continue

                data = source_object[:]

                # Every point-aligned dataset receives
                # exactly the same mask.
                if (
                    data.ndim > 0
                    and data.shape[0] == n_points_before
                ):
                    output_data = data[mask]
                else:
                    output_data = data

                output_dataset = (
                    target_steps.create_dataset(
                        name,
                        data=output_data,
                        dtype=source_object.dtype,
                    )
                )

                # Preserve dataset attributes.
                for key, value in (
                    source_object.attrs.items()
                ):
                    output_dataset.attrs[key] = value

    print(
        f"     Time cut       : "
        f"t <= {time_cut:g} ns"
    )
    print(
        f"     Points         : "
        f"{n_points_before} -> {n_points_after}"
    )
    print(
        f"     Energy         : "
        f"{energy_before:.8g} -> {energy_after:.8g}"
    )
    print(
        f"     Removed energy : "
        f"{energy_before - energy_after:.8g}"
    )
    print(f"     Output         : {output_file}")

    return output_file


# =========================
# safe subprocess
# =========================
def run_cmd(cmd):
    env = os.environ.copy()
    env["ROOT_ENABLE_IMT"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["TBB_NUM_THREADS"] = "1"
    env["XRD_RUNFORKHANDLER"] = "1"

    print("\n[RUN]")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True, env=env)
    time.sleep(2)


# =========================
# split
# =========================
def run_split(root_file):
    em_out = root_file.replace(".root", "_em.root")
    had_out = root_file.replace(".root", "_hadronic.root")

    if Path(em_out).exists() and Path(had_out).exists():
        print("     Skipping Splitting...")
    else:
        run_cmd(["python", "remove_em.py", root_file])
        run_cmd(["python", "remove_hadronic.py", root_file])

    em_out_h5 = root_file.replace(".root", "_em.h5")
    had_out_h5 = root_file.replace(".root", "_hadronic.h5")

    if Path(em_out_h5).exists() and Path(had_out_h5).exists():
        print("     Skipping root2h5...")
        return em_out_h5, had_out_h5

    if not Path(em_out_h5).exists():
        run_cmd(["python", str(ROOT2H5), "--input", em_out])
    if not Path(had_out_h5).exists():
        run_cmd(["python", str(ROOT2H5), "--input", had_out])

    return em_out_h5, had_out_h5


# =========================
# run step2point
# =========================
def run_step2point(
    input_file,
    energy,
    detector,
    component,
    algorithm,
    time_tag,
    extra_args=None,
):
    outdir = Path(
        f"./{detector}/{energy}/outputs/"
        f"{time_tag}/{component}/pipeline_out_{algorithm}"
    )

    output_file = outdir / f"compressed_{algorithm}.h5"

    if output_file.exists():
        print(f"     Skipping {component} {algorithm}...")
        return str(output_file)

    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(STEP2POINT),
        "--input", input_file,
        "--compact-xml", COMPACT_XML[detector],
        "--algorithm", algorithm,
        "--output", str(outdir),
        "--axis", "0", "1", "0",
    ]

    if extra_args:
        cmd.extend(extra_args)

    run_cmd(cmd)
    return str(output_file)


def run_all(input_file, energy, detector, component, time_tag):
    results = {}

    for algorithm in ALGORITHMS:
        try:
            results[algorithm] = run_step2point(
                input_file,
                energy,
                detector,
                component,
                algorithm,
                time_tag,
                EXTRA_ARGS[algorithm],
            )
        except subprocess.CalledProcessError as error:
            print(f"[FAILED] {component} {algorithm}: {error}")
        finally:
            gc.collect()
            time.sleep(2)

    return results


def run_all_hadronic(input_file, energy, detector, component, time_tag):
    results = {}

    for algorithm in ALGORITHMS:
        try:
            results[algorithm] = run_step2point(
                input_file,
                energy,
                detector,
                component,
                algorithm,
                time_tag,
                EXTRA_ARGS_HADRONIC[algorithm],
            )
        except subprocess.CalledProcessError as error:
            print(f"[FAILED] {component} {algorithm}: {error}")
        finally:
            gc.collect()
            time.sleep(2)

    return results


# =========================
# merge
# =========================
def merge(em_file, had_file, out_file):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(em_file, "r") as f1, \
         h5py.File(had_file, "r") as f2, \
         h5py.File(out_file, "w") as fout:

        em_steps = f1["steps"]
        had_steps = f2["steps"]

        # Read datasets once
        em_event = em_steps["event_id"][:]
        had_event = had_steps["event_id"][:]

        em_energy = em_steps["energy"][:]
        had_energy = had_steps["energy"][:]

        em_position = em_steps["position"][:]
        had_position = had_steps["position"][:]

        em_unique = np.unique(em_event)
        had_unique = np.unique(had_event)

        # Some events may disappear from one component after the time cut.
        # Therefore merge over the union instead of requiring exact equality.
        all_events = np.union1d(em_unique, had_unique)

        only_in_em = np.setdiff1d(em_unique, had_unique)
        only_in_had = np.setdiff1d(had_unique, em_unique)

        print(f"    EM events                 : {len(em_unique)}")
        print(f"    Hadronic events           : {len(had_unique)}")
        print(f"    Events in merged union    : {len(all_events)}")
        print(f"    Events only in EM         : {len(only_in_em)}")
        print(f"    Events only in hadronic   : {len(only_in_had)}")

        if len(only_in_em) > 0:
            print(
                "    First events only in EM   :",
                only_in_em[:20],
            )

        if len(only_in_had) > 0:
            print(
                "    First events only in HAD  :",
                only_in_had[:20],
            )

        # Optional datasets are merged only when present in both files.
        optional_names = [
            "time",
            "pdg",
            "cell_id",
            "track_id",
        ]

        optional_data = {}

        for name in optional_names:
            if name in em_steps and name in had_steps:
                optional_data[name] = {
                    "em": em_steps[name][:],
                    "had": had_steps[name][:],
                    "merged": [],
                }
            elif name in em_steps or name in had_steps:
                print(
                    f"    [WARNING] Dataset '{name}' exists in only one "
                    "input file and will not be written."
                )

        merged_event_id = []
        merged_energy = []
        merged_position = []

        for eid in all_events:
            em_mask = em_event == eid
            had_mask = had_event == eid

            event_id = np.concatenate([
                em_event[em_mask],
                had_event[had_mask],
            ])

            energy = np.concatenate([
                em_energy[em_mask],
                had_energy[had_mask],
            ])

            position = np.concatenate([
                em_position[em_mask],
                had_position[had_mask],
            ], axis=0)

            event_optional = {}

            for name, data in optional_data.items():
                event_optional[name] = np.concatenate([
                    data["em"][em_mask],
                    data["had"][had_mask],
                ])

            # Keep every dataset synchronized if sorting by time.
            if "time" in event_optional:
                order = np.argsort(
                    event_optional["time"],
                    kind="stable",
                )

                event_id = event_id[order]
                energy = energy[order]
                position = position[order]

                for name in event_optional:
                    event_optional[name] = event_optional[name][order]

            merged_event_id.append(event_id)
            merged_energy.append(energy)
            merged_position.append(position)

            for name, values in event_optional.items():
                optional_data[name]["merged"].append(values)

        steps = fout.create_group("steps")

        steps.create_dataset(
            "event_id",
            data=np.concatenate(merged_event_id),
        )
        steps.create_dataset(
            "energy",
            data=np.concatenate(merged_energy),
        )
        steps.create_dataset(
            "position",
            data=np.concatenate(merged_position, axis=0),
        )

        for name, data in optional_data.items():
            steps.create_dataset(
                name,
                data=np.concatenate(data["merged"]),
            )

        # EM and HAD files normally contain the same primary information.
        # Copy it once rather than concatenating and duplicating every primary.
        if "primary" in f1:
            f1.copy("primary", fout)

            if "primary" in f2:
                em_primary_events = f1["primary"]["event_id"][:]
                had_primary_events = f2["primary"]["event_id"][:]

                if not np.array_equal(
                    em_primary_events,
                    had_primary_events,
                ):
                    print(
                        "    [WARNING] EM and HAD primary event IDs differ; "
                        "the EM primary group was copied."
                    )

        elif "primary" in f2:
            f2.copy("primary", fout)

    print(f"[OK] merged -> {out_file}")


def run_merges(em_results, had_results, energy, detector):
    merged_results = {}
    merged_dir = Path(f"./{detector}/{energy}/outputs/merged")

    # Combination 1: EM compressed with algorithm A
    #                + hadronic compressed with the same algorithm A.
    for algorithm in ALGORITHMS:
        if algorithm not in em_results or algorithm not in had_results:
            continue

        out_file = merged_dir / (
            f"{detector}_{energy}_{algorithm}_em_{algorithm}_hadronic.h5"
        )
        merge(em_results[algorithm], had_results[algorithm], str(out_file))
        merged_results[f"both_{algorithm}"] = str(out_file)

    # Combination 2: EM identity + hadronic compressed with algorithm A.
    # identity + identity was already produced by Combination 1.
    if "identity" in em_results:
        for algorithm in ALGORITHMS:
            if algorithm == "identity" or algorithm not in had_results:
                continue

            out_file = merged_dir / (
                f"{detector}_{energy}_identity_em_{algorithm}_hadronic.h5"
            )
            merge(em_results["identity"], had_results[algorithm], str(out_file))
            merged_results[f"hadronic_only_{algorithm}"] = str(out_file)
    
    # Combination 3: EM compressed with algorithm A + hadronic identity.
    if "identity" in had_results:
        for algorithm in ALGORITHMS:
            if algorithm == "identity" or algorithm not in em_results:
                continue

            out_file = merged_dir / (
                f"{detector}_{energy}_{algorithm}_em_identity_hadronic.h5"
            )
            merge(em_results[algorithm], had_results["identity"], str(out_file))
            merged_results[f"em_only_{algorithm}"] = str(out_file)

    return merged_results


# =========================
# main pipeline
# =========================
def process_one(energy, detector):
    print(f"\n========== {detector} {energy} ==========")

    root_file = (
        f"./{detector}/{energy}/"
        f"{detector}_piminus_{energy}_Y{Y_POS[detector]}mm.root"
    )
    origin_h5 = root_file.replace(".root", ".h5")

    if Path(origin_h5).exists():
        print("     Skipping root2h5...")
    else:
        run_cmd(["python", str(ROOT2H5), "--input", root_file])

    em_h5, had_h5 = run_split(root_file)

    for time_cut in TIME_CUTS:
        time_tag = f"{time_cut:g}ns"

        print(
            f"\n---------- {detector} {energy} "
            f"time cut = {time_tag} ----------"
        )

        original_cut_h5 = apply_time_cut(origin_h5, time_cut)
        em_cut_h5 = apply_time_cut(em_h5, time_cut)
        had_cut_h5 = apply_time_cut(had_h5, time_cut)

        print("     Compressing EM showers")
        em_results = run_all(str(em_cut_h5), energy, detector, "em", time_tag)

        print("     Compressing hadronic showers")
        had_results = run_all_hadronic(str(had_cut_h5), energy, detector, "hadronic", time_tag)

        print("     Start Merging")
        merged_results = run_merges(em_results, had_results, energy, detector)

        print("     Plotting results")
        for label, merged_file in merged_results.items():
            run_cmd([
                "python",
                "inspect_showers.py",
                "--input", origin_h5, merged_file,
                "--labels", "origin", label,
                "--outdir", f"./{detector}/{energy}/plots",
                "--detector", detector,
                "--energy", energy,
                "--tmax", str(time_cut),
                "--label", str(label),
            ])

            # run_cmd([
            #     "python",
            #     "/eos/user/s/siyuch/step2point/examples/render_shower_display.py",
            #     "--input", origin_h5, merged_file,
            #     "--label", "origin", label,
            #     "--out", f"./{detector}/{energy}/plots/{detector}_{energy}_rendered_showers_display_{label}_{time_cut}ns.png",
            #     "--axis", "0", "1", "0",
            # ])


def main():
    for energy in ENERGIES:
        for detector in ["ODD", "CLD"]:
            process_one(energy, detector)


if __name__ == "__main__":
    main()