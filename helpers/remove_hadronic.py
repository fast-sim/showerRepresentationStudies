#!/usr/bin/env python3
import argparse
import os


import podio
from podio.root_io import Reader, Writer
from edm4hep import (
    SimCalorimeterHitCollection,
    CaloHitContributionCollection,
    CaloHitContribution,
)

EM_PDGS = {11, -11, 22}

DEFAULT_COPY_COLLECTIONS = ("MCParticles",)

DEFAULT_CALO_COLLECTIONS = (
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
)


def get_step_length(c):
    """Some EDM4hep versions have stepLength, some older ones may not."""
    try:
        return c.getStepLength()
    except AttributeError:
        return 0.0


def make_tracked_contribution(contrib_col, old_c):
    """
    Create a new CaloHitContribution that is tracked by contrib_col.

    Important:
    - Do NOT create a standalone CaloHitContribution and directly add it to hit.
      That causes: relation points to untracked object.
    - Do NOT rely on create() + setters if ROOT later shows PDG/energy = 0.
    """

    pdg = int(old_c.getPDG())
    energy = float(old_c.getEnergy())
    time = float(old_c.getTime())
    step_pos = old_c.getStepPosition()
    step_length = float(get_step_length(old_c))

    # Best path: create object directly inside the collection with values.
    try:
        return contrib_col.create(
            pdg,
            energy,
            time,
            step_pos,
            step_length,
        )
    except Exception:
        pass

    # Fallback: construct value object, push it into collection,
    # then retrieve the tracked object from the collection.
    new_value = CaloHitContribution(
        pdg,
        energy,
        time,
        step_pos,
        step_length,
    )

    if hasattr(contrib_col, "push_back"):
        contrib_col.push_back(new_value)
    elif hasattr(contrib_col, "append"):
        contrib_col.append(new_value)
    else:
        raise RuntimeError(
            "Cannot add CaloHitContribution to CaloHitContributionCollection: "
            "no usable create(...), push_back(...), or append(...) method."
        )

    # Retrieve the object that is actually owned/tracked by the collection.
    idx = len(contrib_col) - 1

    try:
        return contrib_col[idx]
    except Exception:
        pass

    try:
        return contrib_col.at(idx)
    except Exception:
        pass

    raise RuntimeError(
        "Contribution was added to collection, but could not retrieve tracked object."
    )
    

def filter_hits(hit_collection):
    """
    Keep only EM contributions from a SimCalorimeterHitCollection.

    Returns:
      - filtered SimCalorimeterHitCollection
      - corresponding CaloHitContributionCollection
      - stats
    """
    new_hit_col = SimCalorimeterHitCollection()
    new_contrib_col = CaloHitContributionCollection()

    n_hits_before = 0
    n_hits_after = 0
    n_contrib_before = 0
    n_contrib_after = 0
    n_hadronic_removed = 0

    for hit in hit_collection:
        n_hits_before += 1

        contributions = list(hit.getContributions())
        n_contrib_before += len(contributions)

        filtered_contribs = [
            c for c in contributions
            if c.getPDG() in EM_PDGS
        ]

        n_hadronic_removed += len(contributions) - len(filtered_contribs)

        # Drop hits that only had EM contributions
        if not filtered_contribs:
            continue

        new_hit = new_hit_col.create()

        # Copy hit-level info
        new_hit.setCellID(hit.getCellID())
        new_hit.setPosition(hit.getPosition())

        # Recompute hit energy from remaining non-EM contributions
        new_energy = sum(c.getEnergy() for c in filtered_contribs)
        new_hit.setEnergy(new_energy)

        # Copy non-EM contributions as tracked objects
        for c in filtered_contribs:
            new_c = make_tracked_contribution(new_contrib_col, c)
            new_hit.addToContributions(new_c)

        n_hits_after += 1
        n_contrib_after += len(filtered_contribs)

    stats = {
        "hits_before": n_hits_before,
        "hits_after": n_hits_after,
        "contrib_before": n_contrib_before,
        "contrib_after": n_contrib_after,
        "hadronic_removed": n_hadronic_removed,
    }

    return new_hit_col, new_contrib_col, stats


def put_collection(frame_out, col, name):
    try:
        frame_out.put(col.clone(), name)
    except Exception:
        frame_out.put(col, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input EDM4hep ROOT file")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output EDM4hep ROOT file with EM contributions removed",
    )
    parser.add_argument(
        "--copy-collections",
        nargs="+",
        default=list(DEFAULT_COPY_COLLECTIONS),
        help="Collections to copy without filtering, e.g. MCParticles",
    )
    parser.add_argument(
        "--calo-collections",
        nargs="+",
        default=list(DEFAULT_CALO_COLLECTIONS),
        help="SimCalorimeterHit collections to filter",
    )
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_em.root"

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Copy collections: {args.copy_collections}")
    print(f"Calo collections to filter: {args.calo_collections}")

    reader = Reader(args.input)
    writer = Writer(args.output)

    for iev, frame_in in enumerate(reader.get("events")):
        frame_out = podio.Frame()
        print(f"\nEvent {iev}")

        # 1. Copy MCParticles unchanged
        for col_name in args.copy_collections:
            try:
                col_in = frame_in.get(col_name)
            except Exception:
                print(f"  skip missing copy collection: {col_name}")
                continue

            print(f"  copying collection unchanged: {col_name}, n={len(col_in)}")
            put_collection(frame_out, col_in, col_name)

        # 2. Filter calorimeter hits
        for col_name in args.calo_collections:
            try:
                col_in = frame_in.get(col_name)
            except Exception:
                print(f"  skip missing calo collection: {col_name}")
                continue

            print(f"  filtering collection: {col_name}, n_hits={len(col_in)}")

            filtered_hit_col, filtered_contrib_col, stats = filter_hits(col_in)

            print(
                f"    hits: {stats['hits_before']} -> {stats['hits_after']}, "
                f"contrib: {stats['contrib_before']} -> {stats['contrib_after']}, "
                f"removed Hadronic contrib: {stats['hadronic_removed']}"
            )

            frame_out.put(filtered_hit_col, col_name)
            frame_out.put(filtered_contrib_col, f"{col_name}Contributions")

        writer.write_frame(frame_out, "events")

    del writer
    print(f"\nWrote EDM4hep ROOT file with EM hit contributions removed: {args.output}")


if __name__ == "__main__":
    main()