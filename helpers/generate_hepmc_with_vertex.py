#!/usr/bin/env python3
"""
Generate HepMC3 AsciiV3 files with explicit vertex information for ddsim.
"""

import argparse
import math

import numpy as np
import pyhepmc


PDG_MASS_GEV = {
    11: 0.000511,
    -11: 0.000511,
    22: 0.0,
    211: 0.13957,
    -211: 0.13957,
    13: 0.10566,
    -13: 0.10566,
}


def normalize_direction(dx, dy, dz):
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0:
        raise ValueError("Direction vector cannot be zero.")
    return dx / norm, dy / norm, dz / norm


def get_position(args, rng, x_range, y_range):
    fixed_values = [args.fixed_x, args.fixed_y, args.fixed_z]

    if any(v is not None for v in fixed_values):
        if not all(v is not None for v in fixed_values):
            raise ValueError(
                "If using fixed position, provide all of "
                "--fixed-x, --fixed-y, and --fixed-z."
            )
        return float(args.fixed_x), float(args.fixed_y), float(args.fixed_z)

    x = rng.uniform(x_range[0], x_range[1])
    y = rng.uniform(y_range[0], y_range[1])
    z = args.z_front
    return float(x), float(y), float(z)


def sample_energy(rng, args, fixed_energy=None):
    if fixed_energy is None:
        fixed_energy = args.energy

    if args.energy_mode == "fixed":
        return float(fixed_energy)

    if args.energy_mode == "uniform":
        return float(rng.uniform(args.emin, args.emax))

    if args.energy_mode == "log_uniform":
        if args.emin <= 0 or args.emax <= 0:
            raise ValueError("emin and emax must be positive for log_uniform.")
        return float(np.exp(rng.uniform(np.log(args.emin), np.log(args.emax))))

    raise ValueError(f"Unknown energy mode: {args.energy_mode}")


def direction_from_axis_angle(axis, theta_deg, phi_deg=0.0):
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    axis = axis.lower()

    if axis == "+z":
        dx = math.sin(theta) * math.cos(phi)
        dy = math.sin(theta) * math.sin(phi)
        dz = math.cos(theta)
    elif axis == "-z":
        dx = math.sin(theta) * math.cos(phi)
        dy = math.sin(theta) * math.sin(phi)
        dz = -math.cos(theta)
    elif axis == "+y":
        dx = math.sin(theta) * math.cos(phi)
        dy = math.cos(theta)
        dz = math.sin(theta) * math.sin(phi)
    elif axis == "-y":
        dx = math.sin(theta) * math.cos(phi)
        dy = -math.cos(theta)
        dz = math.sin(theta) * math.sin(phi)
    elif axis == "+x":
        dx = math.cos(theta)
        dy = math.sin(theta) * math.cos(phi)
        dz = math.sin(theta) * math.sin(phi)
    elif axis == "-x":
        dx = -math.cos(theta)
        dy = math.sin(theta) * math.cos(phi)
        dz = math.sin(theta) * math.sin(phi)
    else:
        raise ValueError("--angle-axis must be one of +x, -x, +y, -y, +z, -z")

    return normalize_direction(dx, dy, dz)


def sample_direction_around_axis(rng, axis, max_theta_deg):
    if max_theta_deg == 0:
        return direction_from_axis_angle(axis, 0.0, 0.0)

    max_theta = math.radians(max_theta_deg)
    cos_theta = rng.uniform(math.cos(max_theta), 1.0)
    theta_deg = math.degrees(math.acos(cos_theta))
    phi_deg = math.degrees(rng.uniform(0.0, 2.0 * math.pi))
    return direction_from_axis_angle(axis, theta_deg, phi_deg)


def get_direction(args, rng):
    vals = [args.dir_x, args.dir_y, args.dir_z]
    if any(v is not None for v in vals):
        if not all(v is not None for v in vals):
            raise ValueError(
                "If using fixed direction, provide all of "
                "--dir-x, --dir-y, and --dir-z."
            )
        return normalize_direction(args.dir_x, args.dir_y, args.dir_z)

    return sample_direction_around_axis(rng, args.angle_axis, args.max_theta_deg)


def momentum_from_energy_direction(pdg_id, energy, direction):
    mass = PDG_MASS_GEV.get(pdg_id, 0.0)
    if energy < mass:
        raise ValueError(
            f"Energy {energy} GeV is smaller than mass {mass} GeV for PDG {pdg_id}."
        )

    p_abs = math.sqrt(max(energy * energy - mass * mass, 0.0))
    dx, dy, dz = direction

    return (
        float(p_abs * dx),
        float(p_abs * dy),
        float(p_abs * dz),
        float(energy),
    )


def make_event(event_number, pdg_id, energy, position, direction, vertex_t):
    event = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
    event.event_number = int(event_number)

    x, y, z = position
    px, py, pz, e = momentum_from_energy_direction(pdg_id, energy, direction)

    # Add a beam/bookkeeping particle first.
    # This helps pyhepmc write a non-empty vertex particle list in AsciiV3.
    incoming = pyhepmc.GenParticle((px, py, pz, e), int(pdg_id), 4)
    outgoing = pyhepmc.GenParticle((px, py, pz, e), int(pdg_id), 1)

    vertex = pyhepmc.GenVertex((x, y, z, float(vertex_t)))
    vertex.add_particle_in(incoming)
    vertex.add_particle_out(outgoing)

    event.add_vertex(vertex)

    return event


def strip_hepmc_extension(filename):
    for ext in [".hepmc", ".hepmc2", ".hepmc3", ".dat", ".txt"]:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return filename


def write_events(output_file, n_events, args, rng, x_range, y_range, direction, fixed_energy=None):
    with pyhepmc.open(output_file, "w", format="hepmc3") as writer:
        for i in range(n_events):
            position = get_position(args, rng, x_range, y_range)
            energy = sample_energy(rng, args, fixed_energy=fixed_energy)
            event = make_event(
                event_number=i + 1,
                pdg_id=args.pdg_id,
                energy=energy,
                position=position,
                direction=direction,
                vertex_t=args.vertex_t,
            )
            writer.write(event)


def generate_single_file(args, rng, x_range, y_range):
    direction = get_direction(args, rng)
    write_events(args.output, args.n_events, args, rng, x_range, y_range, direction)

    print(f"Wrote {args.n_events} HepMC3 AsciiV3 events to {args.output}")
    print("Configuration:")
    print(f"  PDG ID: {args.pdg_id}")
    print(f"  Energy mode: {args.energy_mode}")
    print(f"  Energy: {args.energy} GeV")
    print(f"  Direction: {direction}")
    print(f"  Vertex time: {args.vertex_t}")
    if args.fixed_x is not None:
        print(f"  Fixed vertex: ({args.fixed_x}, {args.fixed_y}, {args.fixed_z}, {args.vertex_t}) mm")
    else:
        print(f"  Vertex x range: {x_range} mm")
        print(f"  Vertex y range: {y_range} mm")
        print(f"  Vertex z: {args.z_front} mm")


def generate_angle_scan(args, rng, x_range, y_range):
    output_base = strip_hepmc_extension(args.output)

    for theta_deg in args.angles:
        output_file = f"{output_base}_theta{theta_deg:g}deg.hepmc"
        direction = direction_from_axis_angle(args.angle_axis, theta_deg, args.phi_deg)
        write_events(
            output_file,
            args.events_per_angle,
            args,
            rng,
            x_range,
            y_range,
            direction,
        )

        print(
            f"Wrote {args.events_per_angle} events at theta={theta_deg:g} deg "
            f"around {args.angle_axis}, phi={args.phi_deg:g} deg to {output_file}"
        )


def generate_energy_scan(args, rng, x_range, y_range):
    output_base = strip_hepmc_extension(args.output)
    direction = direction_from_axis_angle(args.angle_axis, args.theta_deg, args.phi_deg)

    for energy in args.energies:
        output_file = f"{output_base}_energy{energy:g}GeV.hepmc"
        write_events(
            output_file,
            args.events_per_energy,
            args,
            rng,
            x_range,
            y_range,
            direction,
            fixed_energy=energy,
        )

        print(
            f"Wrote {args.events_per_energy} events at energy={energy:g} GeV "
            f"to {output_file}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate true HepMC3 AsciiV3 files with explicit vertex information."
    )

    parser.add_argument("-- ", default="particles.hepmc")
    parser.add_argument("--n-events", type=int, default=100)
    parser.add_argument("--pdg-id", type=int, default=22)
    parser.add_argument("--seed", type=int, default=12345)

    # Default vertex is origin.
    parser.add_argument("--z-front", type=float, default=0.0)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=0.0)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=0.0)

    parser.add_argument("--fixed-x", type=float, default=None)
    parser.add_argument("--fixed-y", type=float, default=None)
    parser.add_argument("--fixed-z", type=float, default=None)
    parser.add_argument("--vertex-t", type=float, default=0.0)

    parser.add_argument(
        "--energy-mode",
        choices=["fixed", "uniform", "log_uniform"],
        default="fixed",
    )
    parser.add_argument("--energy", type=float, default=10.0)
    parser.add_argument("--emin", type=float, default=1.0)
    parser.add_argument("--emax", type=float, default=100.0)

    parser.add_argument("--max-theta-deg", type=float, default=0.0)
    parser.add_argument("--angle-axis", choices=["+x", "-x", "+y", "-y", "+z", "-z"], default="+z")
    parser.add_argument("--dir-x", type=float, default=None)
    parser.add_argument("--dir-y", type=float, default=None)
    parser.add_argument("--dir-z", type=float, default=None)

    parser.add_argument("--angle-scan", action="store_true")
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 5, 10, 15, 20, 30])
    parser.add_argument("--events-per-angle", type=int, default=100)
    parser.add_argument("--phi-deg", type=float, default=0.0)

    parser.add_argument("--energy-scan", action="store_true")
    parser.add_argument("--energies", type=float, nargs="+", default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--events-per-energy", type=int, default=100)
    parser.add_argument("--theta-deg", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    if args.angle_scan:
        generate_angle_scan(args, rng, x_range, y_range)
    elif args.energy_scan:
        generate_energy_scan(args, rng, x_range, y_range)
    else:
        generate_single_file(args, rng, x_range, y_range)


if __name__ == "__main__":
    main()
