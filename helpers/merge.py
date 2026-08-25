from pathlib import Path
import h5py
import numpy as np

STEP2POINT = Path("")
ROOT2H5 = Path("")

def merge(
    em_file,
    had_file,
    out_file,
    axis=(0.0, 1.0, 0.0),
):
    """
    Merge complementary EM and HAD step2point HDF5 files.
    """
    em_file = Path(em_file)
    had_file = Path(had_file)
    out_file = Path(out_file)

    if not em_file.is_file():
        raise FileNotFoundError(f"EM file not found: {em_file}")

    if not had_file.is_file():
        raise FileNotFoundError(f"HAD file not found: {had_file}")

    axis = np.asarray(axis, dtype=np.float64)

    if axis.shape != (3,):
        raise ValueError(
            f"axis must contain exactly three values, got {axis}"
        )

    axis_norm = np.linalg.norm(axis)

    if not np.isfinite(axis_norm) or axis_norm == 0:
        raise ValueError(f"Invalid shower axis: {axis}")

    axis = axis / axis_norm

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with (
        h5py.File(em_file, "r") as f_em,
        h5py.File(had_file, "r") as f_had,
        h5py.File(out_file, "w") as fout,
    ):
        # ------------------------------------------------------------
        # Validate steps groups and fields
        # ------------------------------------------------------------
        if "steps" not in f_em:
            raise KeyError(f"Missing steps group in {em_file}")

        if "steps" not in f_had:
            raise KeyError(f"Missing steps group in {had_file}")

        required_fields = {
            "event_id",
            "energy",
            "position",
        }

        em_fields = set(f_em["steps"].keys())
        had_fields = set(f_had["steps"].keys())

        missing_em = required_fields - em_fields
        missing_had = required_fields - had_fields

        if missing_em:
            raise KeyError(
                f"Missing EM steps fields: {sorted(missing_em)}"
            )

        if missing_had:
            raise KeyError(
                f"Missing HAD steps fields: {sorted(missing_had)}"
            )

        if em_fields != had_fields:
            raise ValueError(
                "EM and HAD steps fields are different:\n"
                f"  EM:  {sorted(em_fields)}\n"
                f"  HAD: {sorted(had_fields)}"
            )

        step_fields = sorted(em_fields)

        # Read input datasets once
        em_steps = {
            key: f_em["steps"][key][:]
            for key in step_fields
        }
        had_steps = {
            key: f_had["steps"][key][:]
            for key in step_fields
        }

        n_em = len(em_steps["event_id"])
        n_had = len(had_steps["event_id"])

        # All fields must have matching first dimensions
        for key in step_fields:
            if len(em_steps[key]) != n_em:
                raise ValueError(
                    f"EM steps/{key} has length "
                    f"{len(em_steps[key])}, expected {n_em}"
                )

            if len(had_steps[key]) != n_had:
                raise ValueError(
                    f"HAD steps/{key} has length "
                    f"{len(had_steps[key])}, expected {n_had}"
                )

        if em_steps["position"].ndim != 2:
            raise ValueError(
                "EM steps/position must be a two-dimensional array"
            )

        if had_steps["position"].ndim != 2:
            raise ValueError(
                "HAD steps/position must be a two-dimensional array"
            )

        if em_steps["position"].shape[1] != 3:
            raise ValueError(
                "EM steps/position must have shape (N, 3)"
            )

        if had_steps["position"].shape[1] != 3:
            raise ValueError(
                "HAD steps/position must have shape (N, 3)"
            )

        em_event = em_steps["event_id"]
        had_event = had_steps["event_id"]

        em_unique = np.unique(em_event)
        had_unique = np.unique(had_event)

        if not np.array_equal(em_unique, had_unique):
            only_em = np.setdiff1d(em_unique, had_unique)
            only_had = np.setdiff1d(had_unique, em_unique)

            raise ValueError(
                "Event IDs are not aligned.\n"
                f"Only in EM:  {only_em[:20]}\n"
                f"Only in HAD: {only_had[:20]}"
            )

        # Merge and sort each event
        merged = {
            key: []
            for key in step_fields
        }

        for eid in em_unique:
            em_mask = em_event == eid
            had_mask = had_event == eid

            event_data = {}

            for key in step_fields:
                event_data[key] = np.concatenate(
                    [
                        em_steps[key][em_mask],
                        had_steps[key][had_mask],
                    ],
                    axis=0,
                )

            n_event_steps = len(event_data["event_id"])

            if n_event_steps == 0:
                continue

            position = np.asarray(
                event_data["position"],
                dtype=np.float64,
            )

            projection = position @ axis

            # sort the steps by track_id, cell_id, z, y, x, projection, and TIME (if available)
            sort_keys = []

            if "track_id" in event_data:
                sort_keys.append(event_data["track_id"])

            if "cell_id" in event_data:
                sort_keys.append(event_data["cell_id"])

            sort_keys.extend(
                [
                    position[:, 2],
                    position[:, 1],
                    position[:, 0],
                    projection,
                ]
            )

            if "time" in event_data:
                sort_keys.append(event_data["time"])
            else:
                sort_keys = [
                    position[:, 2],
                    position[:, 1],
                    position[:, 0],
                    projection,
                ]

            order = np.lexsort(tuple(sort_keys))

            for key in step_fields:
                merged[key].append(event_data[key][order])

        # Write steps group
        steps_out = fout.create_group("steps")

        for key in step_fields:
            if not merged[key]:
                raise ValueError(
                    f"No values were merged for steps/{key}"
                )

            output_data = np.concatenate(
                merged[key],
                axis=0,
            )

            steps_out.create_dataset(
                key,
                data=output_data,
            )

        output_length = len(steps_out["event_id"])

        for key in steps_out.keys():
            if len(steps_out[key]) != output_length:
                raise RuntimeError(
                    f"Output steps/{key} has length "
                    f"{len(steps_out[key])}, "
                    f"expected {output_length}"
                )

        # Check and copy primary group once
        em_has_primary = "primary" in f_em
        had_has_primary = "primary" in f_had

        if em_has_primary != had_has_primary:
            raise ValueError(
                "The primary group exists in only one input file"
            )

        if em_has_primary:
            em_primary_fields = set(f_em["primary"].keys())
            had_primary_fields = set(f_had["primary"].keys())

            if em_primary_fields != had_primary_fields:
                raise ValueError(
                    "EM and HAD primary fields differ:\n"
                    f"  EM:  {sorted(em_primary_fields)}\n"
                    f"  HAD: {sorted(had_primary_fields)}"
                )

            for key in sorted(em_primary_fields):
                em_data = f_em["primary"][key][:]
                had_data = f_had["primary"][key][:]

                try:
                    identical = np.array_equal(
                        em_data,
                        had_data,
                        equal_nan=True,
                    )
                except TypeError:
                    identical = np.array_equal(
                        em_data,
                        had_data,
                    )

                if not identical:
                    raise ValueError(
                        f"EM and HAD primary/{key} "
                        "are not identical"
                    )

            f_em.copy("primary", fout)

        # Copy metadata once
        em_has_metadata = "metadata" in f_em
        had_has_metadata = "metadata" in f_had

        if em_has_metadata != had_has_metadata:
            raise ValueError(
                "The metadata group exists in only one input file"
            )

        if em_has_metadata:
            f_em.copy("metadata", fout)

    print(
        f"[OK] merged {n_em} EM steps + "
        f"{n_had} HAD steps = {output_length} steps"
    )
    print(f"[OK] shower axis = {axis}")
    print(f"[OK] output → {out_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge EM and HAD HDF5 files.")
    parser.add_argument("em_file", type=str, help="Path to the EM HDF5 file.")
    parser.add_argument("had_file", type=str, help="Path to the HAD HDF5 file.")
    parser.add_argument(
        "out_file", 
        type=str, 
        default="./merged.h5", 
        help="Path to the output merged HDF5 file."
    )

    args = parser.parse_args()

    merge(
        args.em_file, 
        args.had_file, 
        args.out_file,
        axis=(0.0, 1.0, 0.0),
    )


if __name__ == "__main__":
    main()