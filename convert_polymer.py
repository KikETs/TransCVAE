#!/usr/bin/env python3
"""
.mol/.mol2  ➜  *.car / *.mdf   ➜  (optional) msi2lmp
PCFF+ / Class-II        2025-06-24

Requires: RDKit ≥2023.09
"""

from pathlib import Path
import argparse, sys, shutil, subprocess, math, textwrap, datetime
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import combinations

# ────────────────────────────────────────── 유틸리티 ──────────────────────────────────────────
def _which(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        sys.exit(f"[ERROR] '{cmd}' not found in PATH")
    return path

def _run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def safe_charge(atom):
    if atom.HasProp('_GasteigerCharge'):
        q = atom.GetDoubleProp('_GasteigerCharge')
        return 0.0 if math.isnan(q) else q
    return 0.0

def full_name(atom, idx, seg="SYS"):
    """Return the full Materials-Studio atom name."""
    return f"{seg}_1:{atom.GetSymbol()}{idx}"

def atom_label(at: Chem.Atom, idx: int) -> str:
    """Bare label: element symbol + index (1-based)."""
    return f"{at.GetSymbol()}{idx}"
# ────────────────────────────── PCFF 타입 간단 휴리스틱 ──────────────────────────────
from rdkit.Chem.rdchem import HybridizationType as Hyb
def guess_pcff(atom):
    sym   = atom.GetSymbol()
    hyb   = atom.GetHybridization()
    aro   = atom.GetIsAromatic()
    ring3 = atom.IsInRingSize(3)

    # ───────── Carbon ─────────
    if sym == "C":
        if aro:
            return "cp"
        if hyb == Hyb.SP:
            return "ct"        # C≡C
        if hyb == Hyb.SP2:
            return "c="        # 비방향족 sp²
        return "c3"            # sp³

    # ───────── Oxygen ─────────
    if sym == "O":
        return "o_2" if hyb == Hyb.SP2 else "o"   # o: sp³(알코올/에테르)

    # ───────── Nitrogen ───────
    if sym == "N":
        if aro:
            return "nb"                         # 방향족 아민
        if hyb == Hyb.SP2:
            return "n2"                         # sp²(아마이드 등)
        if ring3:
            return "n3m"                        # 3-membered ring
        return "na"                             # 일반 sp³ 아민

    # ───────── Sulfur ─────────
    if sym == "S":
        return "sp" if aro else "s"             # sp³ S vs 방향족 S

    # ───────── Hydrogen ───────
    if sym == "H":
        # N/O에 붙은 H → h*, 그 외 → h
        heavy = atom.GetBonds()[0].GetOtherAtom(atom).GetSymbol()
        return "h*" if heavy in ("N", "O") else "h"

    # 기타 원소: 소문자 심벌 그대로
    return sym.lower()


# ───────────────────────────────── CAR / MDF 라이팅 ─────────────────────────────────
# ─────────────────────────── CAR ───────────────────────────
def write_car(mol: Chem.Mol, fname: Path, seg="XXXX"):
    cell: tuple = (40.0, 40.0, 40.0, 90.0, 90.0, 90.0)
    a, b, c, alpha, beta, gamma = cell
    today = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    conf = mol.GetConformer()
    lines = ["!BIOSYM archive 3", "PBC=ON",
             "Materials Studio Generated CAR File",
             f"!DATE: {today}   Materials Studio Generated MDF file",
             f"PBC{a:10.4f}{b:10.4f}{c:10.4f}{alpha:8.2f}{beta:8.2f}{gamma:8.2f}", ""]
    for i, at in enumerate(mol.GetAtoms(), start=1):
        name = atom_label(at, i)                # e.g. "H1", "O2"
        x, y, z = conf.GetAtomPosition(i-1)
        lines.append(
            f"{name:<8}{x:10.4f}{y:10.4f}{z:10.4f} "
            f"{seg:<4} 1 {guess_pcff(at):<4} {at.GetSymbol():<2} {safe_charge(at):6.4f}"
        )
    lines += ["end", "end"]
    fname.write_text("\n".join(lines))

# ─────────────────────────── MDF ───────────────────────────
def write_mdf(mol: Chem.Mol, fname: Path, molname="poly", seg="XXXX"):
    if not mol.GetAtomWithIdx(0).HasProp('_GasteigerCharge'):
        AllChem.ComputeGasteigerCharges(mol)

    # neighbor list
    neigh = {i: [n.GetIdx() for n in at.GetNeighbors()]
             for i, at in enumerate(mol.GetAtoms())}

    today = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    hdr = [
        "!BIOSYM molecular_data 4", "",
        f"!Date: {today}   Materials Studio Generated MDF file", "",
        "#topology", "",
        "@column 1 element", "@column 2 atom_type", "@column 3 charge_group",
        "@column 4 isotope", "@column 5 formal_charge", "@column 6 charge",
        "@column 7 switching_atom", "@column 8 oop_flag",
        "@column 9 chirality_flag", "@column 10 occupancy",
        "@column 11 xray_temp_factor", "@column 12 connections", "",
        f"@molecule {molname}", ""
    ]

    body = []
    for i, at in enumerate(mol.GetAtoms(), start=1):
        name = full_name(at, i, seg)
        #name = atom_label(at, i)                # same bare label
        q    = safe_charge(at)
        base = (f"{name:<15}{at.GetSymbol():<2}{guess_pcff(at):<4}"
                f" 1 0 0 {q:8.4f} 0 0 8 1.0000  0.0000 ")
        conns = " ".join(f"{atom_label(mol.GetAtomWithIdx(j), j+1)}"
                         for j in neigh[i-1])
        for line in (base + conns):
            body.append(line)

    tail = ["", "!", "#symmetry", "@periodicity 3 xyz", "@group (P1)", "", "#end"]
    fname.write_text("\n".join(hdr + body + tail))


# ─────────────────────────────────────────── 메인 ───────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Generate .car/.mdf (PCFF+) from .mol/.mol2")
    p.add_argument("input", help="polymer.mol or polymer.mol2")
    p.add_argument("--prefix", help="output basename (default: input stem)")
    p.add_argument("--run-msi2lmp", action="store_true", help="invoke msi2lmp afterwards")
    p.add_argument("--frc", default="pcff", help="*.frc file (default: pcff)")
    p.add_argument("--no-topology", action="store_true",
              help="skip BOND/ANGLE/DIHEDRAL/IMPROPER blocks")
    args = p.parse_args()

    infile = Path(args.input).resolve()
    if not infile.exists(): sys.exit(f"{infile} not found")

    stem = args.prefix or infile.stem
    if infile.suffix.lower() == ".mol2":
        mol = Chem.MolFromMol2File(str(infile), sanitize=False, removeHs=False)
    else:
        mol = Chem.MolFromMolFile(str(infile), sanitize=False, removeHs=False)
    if mol is None:
        sys.exit("RDKit failed to parse the structure.")

    # --- RDKit 버전별 호환 처리 ---
    if hasattr(Chem, "AssignAtomChiralTags"):
        Chem.AssignAtomChiralTags(mol)
    elif hasattr(Chem.rdmolops, "AssignAtomChiralTagsFromStructure"):
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    else:
        print("[WARN] RDKit 버전이 낮아 키랄 태그 자동 지정 기능이 없습니다 — 건너뜀.")
    AllChem.Compute2DCoords(mol) if not mol.GetConformers() else None
    AllChem.EmbedMolecule(mol, randomSeed=1) if not mol.GetConformers() else None
    AllChem.ComputeGasteigerCharges(mol)

    car = Path(f"{stem}.car");  mdf = Path(f"{stem}.mdf")
    write_car(mol, car);  write_mdf(mol, mdf)
    print(f"[OK] wrote {car} and {mdf}")

    if args.run_msi2lmp:
        msi2lmp = _which("msi2lmp.exe")
        _run([msi2lmp, stem, "-class", "2", "-f", args.frc])    # Class-II flag

if __name__ == "__main__":
    main()
