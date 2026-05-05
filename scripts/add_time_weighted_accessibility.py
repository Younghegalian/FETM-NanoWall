#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from numpy.lib import format as npy_format


DEFAULT_V_MEAN_UM_S = 370353425.4688162
KCR_FIELD = "kinetic_contact_rate_s_inv"
LEGACY_FIELD = "time_weighted_accessibility_s_inv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add kinetic contact rate KCR = (v_mean/lambda) * accessibility to transport_fields.npz."
    )
    parser.add_argument("transport", nargs="+", help="transport_fields.npz path(s).")
    parser.add_argument("--v-mean-um-s", type=float, default=DEFAULT_V_MEAN_UM_S)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for item in args.transport:
        path = Path(item)
        add_kinetic_contact_rate_field(path, v_mean_um_s=args.v_mean_um_s)
        print(path)
    return 0


def add_kinetic_contact_rate_field(path: Path, *, v_mean_um_s: float) -> None:
    original_mode = path.stat().st_mode
    with np.load(path, allow_pickle=False) as data:
        meta = _read_metadata(data)
        lambda_um = float(meta["lambda_um"])
        scale_s_inv = float(v_mean_um_s) / lambda_um
        meta["v_mean_um_s"] = float(v_mean_um_s)
        meta["kinetic_contact_rate_scale_s_inv"] = scale_s_inv
        meta["kinetic_contact_rate_definition"] = (
            "kinetic_contact_rate_s_inv = accessibility * v_mean_um_s / lambda_um"
        )
        meta["time_weighted_accessibility_scale_s_inv"] = scale_s_inv
        meta["time_weighted_accessibility_definition"] = (
            "Deprecated alias of kinetic_contact_rate_s_inv."
        )
        metadata_json = np.array([json.dumps(meta, indent=2, sort_keys=True)])

        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                for name in data.files:
                    if name in ("metadata_json", KCR_FIELD, LEGACY_FIELD):
                        continue
                    _write_npy_to_zip(zf, name, np.asarray(data[name]))

                kcr = np.asarray(data["accessibility"], dtype=np.float32) * np.float32(scale_s_inv)
                _write_npy_to_zip(zf, KCR_FIELD, kcr)
                _write_npy_to_zip(zf, LEGACY_FIELD, kcr)
                _write_npy_to_zip(zf, "metadata_json", metadata_json)
            tmp_path.replace(path)
            path.chmod(original_mode & 0o777)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def _read_metadata(data: np.lib.npyio.NpzFile) -> dict:
    raw = np.asarray(data["metadata_json"]).ravel()[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _write_npy_to_zip(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    with zf.open(f"{name}.npy", mode="w", force_zip64=True) as handle:
        npy_format.write_array(handle, arr, allow_pickle=False)


if __name__ == "__main__":
    raise SystemExit(main())
