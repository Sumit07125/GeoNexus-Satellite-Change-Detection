"""
sar_validation.py — Sentinel-1 SAR channel verification and diagnostics for Geo-Nexus.

Provides per-channel physical validation, empirical training distribution
checks, and structured diagnostics for Sentinel-1 C-band backscatter channels:
  - Channel 13: VV (co-polarization, dB × 100)
  - Channel 14: VH (cross-polarization, dB × 100)
  - Channel 15: VH_VV (cross-polarization ratio / difference, dB × 100)

Physical reference:
  - Sentinel-1 GRD backscatter calibrated in dB.
  - Typical rural/semi-urban training quantiles from FINAL_ARCH_3.md §15 and
    norm_stats_trainonly.json:
      VV:    mean = -9.84 dB, std = 4.23 dB (typical ~[-35, +10] dB)
      VH:    mean = -17.24 dB, std = 4.16 dB (typical ~[-40, +5] dB)
      VH_VV: mean = -7.40 dB, std = 2.50 dB (typical ~[-35, +5] dB)
  - Urban dihedral / trihedral corner reflectors (buildings, bridges, industrial
    steel structures) legitimately produce intense co-polarization double-bounce
    backscatter reaching +10 dB to +25 dB (1000 to 2500 in dB×100).
  - Project validation guardrail bounds (operational limits for rejecting corrupted or
    uncalibrated values; these represent pipeline software guardrails rather than
    official ESA instrument specifications):
      VV:    [-6000, 3000] (-60 dB to +30 dB)
      VH:    [-6000, 2000] (-60 dB to +20 dB)
      VH_VV: [-6000, 1500] (-60 dB to +15 dB)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

SAR_CHANNELS = ["VV", "VH", "VH_VV"]
SAR_CHANNEL_INDICES = {"VV": 13, "VH": 14, "VH_VV": 15}

# Project validation guardrail bounds: hard errors if violated (indicates corruption, unphysical values, or sensor saturation)
# Note: These are Geo-Nexus project operational guardrails calibrated for the Sentinel-1 GRD ingestion pipeline,
# not official ESA sensor specifications.
# Stored units: dB × 100
SAR_PHYSICAL_LIMITS = {
    "VV": {"lower": -6000.0, "upper": 3000.0, "unit": "dB_x100"},
    "VH": {"lower": -6000.0, "upper": 2000.0, "unit": "dB_x100"},
    "VH_VV": {"lower": -6000.0, "upper": 1500.0, "unit": "dB_x100"},
}

# Empirical training distribution bounds: soft warning if outside typical rural/semi-urban training quantiles
# (from FINAL_ARCH_3.md §15.1 and norm_stats_trainonly.json)
SAR_TRAINING_WARN_THRESHOLDS = {
    "VV": {"lower": -3500.0, "upper": 1000.0, "unit": "dB_x100"},
    "VH": {"lower": -4000.0, "upper": 500.0, "unit": "dB_x100"},
    "VH_VV": {"lower": -3500.0, "upper": 500.0, "unit": "dB_x100"},
}


def sar_channel_diagnostics(
    image: np.ndarray,
    channel_names: list[str] | None = None,
    physical_limits: dict[str, dict[str, float]] | None = None,
    warn_thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute bounded per-channel statistics and range compliance for SAR channels.

    Accepts either:
      - Full 17-channel array [17, H, W] where channels 13, 14, 15 are VV, VH, VH_VV
      - 3-channel SAR slice [3, H, W] where channels 0, 1, 2 are VV, VH, VH_VV
    """
    if channel_names is None:
        channel_names = SAR_CHANNELS
    if physical_limits is None:
        physical_limits = SAR_PHYSICAL_LIMITS
    if warn_thresholds is None:
        warn_thresholds = SAR_TRAINING_WARN_THRESHOLDS

    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Image must be a 3D numpy array [C, H, W].")

    is_full_tensor = image.shape[0] >= 16
    h, w = image.shape[1], image.shape[2]
    total_px = h * w

    results: dict[str, dict[str, Any]] = {}

    for i, name in enumerate(channel_names):
        slice_idx = 13 + i if is_full_tensor else i
        channel_idx = 13 + i  # Standard 17-channel index
        values = np.asarray(image[slice_idx], dtype=np.float64)
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        nonfinite_count = int((~finite_mask).sum())
        finite_count = int(finite.size)

        limits = physical_limits.get(name, {"lower": -6000.0, "upper": 3000.0})
        warns = warn_thresholds.get(name, {"lower": -3500.0, "upper": 1000.0})

        row: dict[str, Any] = {
            "channel_index": channel_idx,
            "channel": name,
            "units": "dB_x100",
            "shape": list(values.shape),
            "total_pixels": total_px,
            "finite_fraction": float(finite_mask.mean()) if total_px > 0 else 0.0,
            "finite_count": finite_count,
            "nonfinite_count": nonfinite_count,
            "lower_limit": float(limits["lower"]),
            "upper_limit": float(limits["upper"]),
            "training_warn_lower": float(warns["lower"]),
            "training_warn_upper": float(warns["upper"]),
        }

        if finite.size > 0:
            p = np.percentile(finite, [0, 0.1, 1, 5, 50, 95, 99, 99.9, 100])
            obs_min = float(finite.min())
            obs_max = float(finite.max())
            below_phys = int((finite < limits["lower"]).sum())
            above_phys = int((finite > limits["upper"]).sum())
            below_warn = int((finite < warns["lower"]).sum())
            above_warn = int((finite > warns["upper"]).sum())

            # Evaluate spatial distribution of physical anomalies if any exist
            anomalies_mask = (values < limits["lower"]) | (values > limits["upper"])
            boundary_anomaly_count = 0
            interior_anomaly_count = 0
            if anomalies_mask.any() and h > 4 and w > 4:
                border_mask = np.zeros((h, w), dtype=bool)
                border_mask[:2, :] = True
                border_mask[-2:, :] = True
                border_mask[:, :2] = True
                border_mask[:, -2:] = True
                boundary_anomaly_count = int((anomalies_mask & border_mask).sum())
                interior_anomaly_count = int((anomalies_mask & ~border_mask).sum())

            spatial_dist = "none"
            if (below_phys + above_phys) > 0:
                if boundary_anomaly_count > 0 and interior_anomaly_count == 0:
                    spatial_dist = "boundary_only"
                elif interior_anomaly_count > 0 and boundary_anomaly_count == 0:
                    spatial_dist = "interior_only"
                else:
                    spatial_dist = "widespread"

            row.update({
                "min": obs_min,
                "max": obs_max,
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "percentiles": [round(float(v), 2) for v in p],
                "zero_fraction": float((finite == 0).mean()),
                "below_limit_count": below_phys,
                "above_limit_count": above_phys,
                "below_limit_fraction": float(below_phys / total_px),
                "above_limit_fraction": float(above_phys / total_px),
                "below_warn_count": below_warn,
                "above_warn_count": above_warn,
                "spatial_distribution": spatial_dist,
            })
        else:
            row.update({
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "percentiles": [],
                "zero_fraction": 0.0,
                "below_limit_count": 0,
                "above_limit_count": 0,
                "below_limit_fraction": 0.0,
                "above_limit_fraction": 0.0,
                "below_warn_count": 0,
                "above_warn_count": 0,
                "spatial_distribution": "all_nonfinite",
            })

        results[name] = row

    return results


def check_sar_channel_contracts(
    image: np.ndarray,
    label: str = "T1",
    provider: str = "Google Earth Engine",
    date: str | None = None,
    meta: dict | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Validate SAR channels against physical bounds and collect warnings.

    Returns:
      (sar_diagnostics, warnings_to_record)

    Raises:
      ValueError or PipelineError (via exception_factory if provided, or standard ValueError)
      if non-finite pixels or physical contract violations occur.
    """
    diagnostics = sar_channel_diagnostics(image)
    warnings: list[dict[str, Any]] = []

    # 1. Non-finite pixels check
    nonfinite_channels = [name for name, row in diagnostics.items() if row["nonfinite_count"] > 0]
    if nonfinite_channels:
        first = diagnostics[nonfinite_channels[0]]
        err_details = {
            "provider": provider,
            "stage": "sar_channel_range_validation",
            "date": date,
            "label": label,
            "channel": nonfinite_channels[0] if len(nonfinite_channels) == 1 else nonfinite_channels,
            "channel_index": first["channel_index"],
            "units": "dB_x100",
            "nonfinite_count": first["nonfinite_count"],
            "total_pixels": first["total_pixels"],
            "finite_fraction": first["finite_fraction"],
            "all_sar_diagnostics": diagnostics,
        }
        ch_list = ", ".join(nonfinite_channels)
        raise _build_sar_pipeline_error(
            "invalid_input_channels",
            f"{label} Sentinel-1 channel(s) {ch_list} contain NaN or infinite values.",
            err_details,
        )

    # 2. Physical instrument bounds check
    physical_violations: list[dict[str, Any]] = []
    for name, row in diagnostics.items():
        if row["below_limit_count"] > 0 or row["above_limit_count"] > 0:
            physical_violations.append(row)

    if physical_violations:
        first = physical_violations[0]
        violating_names = [v["channel"] for v in physical_violations]
        err_details = {
            "provider": provider,
            "stage": "sar_channel_range_validation",
            "date": date,
            "label": label,
            "channel": first["channel"] if len(physical_violations) == 1 else violating_names,
            "channel_index": first["channel_index"],
            "units": "dB_x100",
            "observed_min": first["min"],
            "observed_max": first["max"],
            "lower_limit": first["lower_limit"],
            "upper_limit": first["upper_limit"],
            "below_limit_count": first["below_limit_count"],
            "above_limit_count": first["above_limit_count"],
            "total_pixels": first["total_pixels"],
            "finite_fraction": first["finite_fraction"],
            "spatial_distribution": first["spatial_distribution"],
            "violating_channels": violating_names,
            "all_sar_diagnostics": diagnostics,
        }
        msg = (
            f"{label} Sentinel-1 {first['channel']} channel is outside physical limits: "
            f"observed [{first['min']:.1f}, {first['max']:.1f}] dB×100 exceeds "
            f"contract limits [{first['lower_limit']:.1f}, {first['upper_limit']:.1f}] "
            f"({first['below_limit_count']} below, {first['above_limit_count']} above out of {first['total_pixels']} px)."
        )
        raise _build_sar_pipeline_error("invalid_input_channels", msg, err_details)

    # 3. Training distribution soft warnings (e.g. urban double-bounce with VV > +10 dB)
    for name, row in diagnostics.items():
        above = row["above_warn_count"]
        below = row["below_warn_count"]
        if above > 0 or below > 0:
            if name == "VV" and above > 0:
                interpretation = (
                    f"High co-polarization backscatter (VV up to {row['max'] / 100.0:.2f} dB, "
                    f"{above} pixels > +10 dB) detected in {label}. "
                    "This is characteristic of urban double-bounce and dihedral/trihedral corner reflections "
                    "from buildings, bridges, and infrastructure. Physical signal is preserved; inference proceeds."
                )
                code = "sar_vv_urban_reflection_notice"
            elif name == "VH_VV" and below > 0:
                interpretation = (
                    f"Strong cross-polarization contrast (VH_VV down to {row['min'] / 100.0:.2f} dB, "
                    f"{below} pixels < -35 dB) detected in {label}. "
                    "Consistent with strong co-pol double-bounce relative to diffuse cross-pol. Physical signal is preserved; inference proceeds."
                )
                code = "sar_vh_vv_contrast_notice"
            else:
                interpretation = (
                    f"{label} {name} has values outside typical training quantiles "
                    f"(min={row['min'] / 100.0:.2f} dB, max={row['max'] / 100.0:.2f} dB; {above + below} pixels outside). "
                    "Values are within physical instrument limits; inference proceeds."
                )
                code = f"sar_{name.lower()}_distribution_notice"

            warnings.append({
                "code": code,
                "channel": name,
                "channel_index": row["channel_index"],
                "label": label,
                "date": date,
                "units": "dB_x100",
                "observed_min": row["min"],
                "observed_max": row["max"],
                "training_warn_lower": row["training_warn_lower"],
                "training_warn_upper": row["training_warn_upper"],
                "outside_count": above + below,
                "total_pixels": row["total_pixels"],
                "percentiles": row["percentiles"],
                "interpretation": interpretation,
            })

    return diagnostics, warnings


def _build_sar_pipeline_error(code: str, message: str, details: dict[str, Any]) -> Exception:
    """Helper to raise PipelineError if imported in app context, otherwise custom ValueError."""
    try:
        from app import PipelineError
        return PipelineError(code, message, details)
    except Exception:
        exc = ValueError(f"[{code}] {message}")
        exc.code = code  # type: ignore[attr-defined]
        exc.details = details  # type: ignore[attr-defined]
        return exc

