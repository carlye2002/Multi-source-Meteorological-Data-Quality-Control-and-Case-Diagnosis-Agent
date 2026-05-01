#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气象多源资料质控与个例诊断 Agent

功能：
1. 自动扫描并接入 NetCDF / GRIB / CSV 气象资料
2. 对多源资料做基础标准化
3. 执行缺测、异常值、时间连续性、空间一致性质控
4. 针对强降水/台风/高温/寒潮等个例生成诊断图表
5. 生成 Markdown 个例诊断报告

建议运行：
    python meteo_multi_source_qc_diagnostic_agent.py \
        --input ./data \
        --output ./outputs \
        --case-name "2024-07-20 强降水个例" \
        --event heavy_rain \
        --start "2024-07-20 00:00" \
        --end "2024-07-21 00:00"

无数据演示：
    python meteo_multi_source_qc_diagnostic_agent.py --make-sample --input ./sample_data --output ./outputs

依赖：
    pip install numpy pandas xarray matplotlib pyyaml netCDF4 scipy

可选依赖：
    pip install cfgrib eccodes
    用于读取 GRIB/GRIB2 文件。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# -----------------------------
# 基础配置
# -----------------------------

@dataclass
class AgentConfig:
    input_dir: Path
    output_dir: Path
    case_name: str = "未命名气象个例"
    event_type: str = "auto"  # auto / heavy_rain / typhoon / heatwave / cold_wave / convection
    start_time: Optional[pd.Timestamp] = None
    end_time: Optional[pd.Timestamp] = None
    region: Optional[Tuple[float, float, float, float]] = None  # lon_min, lon_max, lat_min, lat_max
    variable_alias: Dict[str, List[str]] = field(default_factory=lambda: {
        "precip": ["precip", "pre", "rain", "tp", "total_precipitation", "降水", "降水量", "PRE_1h"],
        "temperature": ["t2m", "tmp", "temp", "temperature", "T", "气温", "温度"],
        "u_wind": ["u", "u10", "u_component_of_wind", "UGRD", "u_wind"],
        "v_wind": ["v", "v10", "v_component_of_wind", "VGRD", "v_wind"],
        "rh": ["rh", "relative_humidity", "humidity", "相对湿度"],
        "mslp": ["mslp", "slp", "prmsl", "mean_sea_level_pressure", "海平面气压"],
        "cape": ["cape", "CAPE"],
        "cin": ["cin", "CIN"],
        "w": ["w", "omega", "vertical_velocity", "垂直速度"],
        "reflectivity": ["ref", "dbz", "reflectivity", "雷达回波"],
    })
    physical_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "precip": (0.0, 1000.0),
        "temperature": (-90.0, 60.0),
        "u_wind": (-100.0, 100.0),
        "v_wind": (-100.0, 100.0),
        "rh": (0.0, 100.0),
        "mslp": (800.0, 1100.0),
        "cape": (0.0, 10000.0),
        "cin": (-2000.0, 1000.0),
        "reflectivity": (-20.0, 80.0),
    })
    max_files: int = 200
    station_value_columns: List[str] = field(default_factory=lambda: [
        "precip", "pre", "rain", "temperature", "temp", "t2m", "wind_speed", "value"
    ])


@dataclass
class DataAsset:
    path: Path
    source_type: str  # netcdf / grib / csv
    dataset: Optional[xr.Dataset] = None
    dataframe: Optional[pd.DataFrame] = None
    canonical_vars: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QCRecord:
    asset: str
    variable: str
    check_name: str
    severity: str  # info / warning / error
    message: str
    count: Optional[int] = None
    ratio: Optional[float] = None


@dataclass
class DiagnosticResult:
    name: str
    text: str
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: List[Path] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# 工具函数
# -----------------------------

def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("meteo_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None or str(value).strip() == "":
        return None
    return pd.to_datetime(value)


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]+", "_", text)
    return text.strip("_")[:120] or "unnamed"


def is_lat_name(name: str) -> bool:
    return name.lower() in {"lat", "latitude", "y", "纬度"}


def is_lon_name(name: str) -> bool:
    return name.lower() in {"lon", "longitude", "x", "经度"}


def is_time_name(name: str) -> bool:
    return name.lower() in {"time", "valid_time", "datetime", "date", "timestamp", "时间"}


def find_first_existing(names: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    name_map = {str(n).lower(): str(n) for n in names}
    for c in candidates:
        if c.lower() in name_map:
            return name_map[c.lower()]
    return None


def canonicalize_vars(names: Iterable[str], alias: Dict[str, List[str]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    lower_to_original = {str(n).lower(): str(n) for n in names}
    for canon, aliases in alias.items():
        for a in aliases:
            if a.lower() in lower_to_original:
                result[canon] = lower_to_original[a.lower()]
                break
    return result


def subset_dataset(ds: xr.Dataset, cfg: AgentConfig) -> xr.Dataset:
    result = ds

    if cfg.start_time is not None or cfg.end_time is not None:
        time_coord = find_first_existing(result.coords, ["time", "valid_time"])
        if time_coord:
            start = cfg.start_time if cfg.start_time is not None else result[time_coord].min().values
            end = cfg.end_time if cfg.end_time is not None else result[time_coord].max().values
            result = result.sel({time_coord: slice(start, end)})

    if cfg.region is not None:
        lon_min, lon_max, lat_min, lat_max = cfg.region
        lat_coord = find_first_existing(result.coords, ["lat", "latitude", "y"])
        lon_coord = find_first_existing(result.coords, ["lon", "longitude", "x"])
        if lat_coord and lon_coord:
            lat_values = result[lat_coord].values
            lon_values = result[lon_coord].values
            lat_slice = slice(lat_min, lat_max) if lat_values[0] <= lat_values[-1] else slice(lat_max, lat_min)
            lon_slice = slice(lon_min, lon_max) if lon_values[0] <= lon_values[-1] else slice(lon_max, lon_min)
            result = result.sel({lat_coord: lat_slice, lon_coord: lon_slice})

    return result


def subset_dataframe(df: pd.DataFrame, cfg: AgentConfig) -> pd.DataFrame:
    out = df.copy()
    time_col = find_first_existing(out.columns, ["time", "datetime", "date", "timestamp", "时间"])
    if time_col:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        if cfg.start_time is not None:
            out = out[out[time_col] >= cfg.start_time]
        if cfg.end_time is not None:
            out = out[out[time_col] <= cfg.end_time]

    if cfg.region is not None:
        lon_min, lon_max, lat_min, lat_max = cfg.region
        lat_col = find_first_existing(out.columns, ["lat", "latitude", "纬度"])
        lon_col = find_first_existing(out.columns, ["lon", "longitude", "经度"])
        if lat_col and lon_col:
            out = out[
                (out[lon_col] >= lon_min) & (out[lon_col] <= lon_max) &
                (out[lat_col] >= lat_min) & (out[lat_col] <= lat_max)
            ]
    return out


def summarize_array(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "valid": 0, "missing_ratio": 1.0}
    return {
        "count": int(values.size),
        "valid": int(finite.size),
        "missing_ratio": float(1 - finite.size / max(values.size, 1)),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "p95": float(np.nanpercentile(finite, 95)),
        "p99": float(np.nanpercentile(finite, 99)),
    }


def ensure_output_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        "base": base,
        "figures": base / "figures",
        "tables": base / "tables",
        "logs": base / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


# -----------------------------
# Agent 1：数据接入
# -----------------------------

class DataIngestionAgent:
    def __init__(self, cfg: AgentConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def scan_files(self) -> List[Path]:
        if not self.cfg.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在：{self.cfg.input_dir}")

        patterns = ["*.nc", "*.nc4", "*.cdf", "*.grib", "*.grb", "*.grb2", "*.csv"]
        files: List[Path] = []
        for pat in patterns:
            files.extend(self.cfg.input_dir.rglob(pat))
        files = sorted(set(files))[: self.cfg.max_files]
        self.logger.info("扫描到 %d 个候选资料文件", len(files))
        return files

    def load_all(self) -> List[DataAsset]:
        assets: List[DataAsset] = []
        for path in self.scan_files():
            try:
                asset = self.load_one(path)
                assets.append(asset)
                self.logger.info("接入成功：%s", path.name)
            except Exception as exc:
                self.logger.warning("接入失败：%s | %s", path, exc)
        return assets

    def load_one(self, path: Path) -> DataAsset:
        suffix = path.suffix.lower()
        if suffix in {".nc", ".nc4", ".cdf"}:
            return self._load_netcdf(path)
        if suffix in {".grib", ".grb", ".grb2"}:
            return self._load_grib(path)
        if suffix == ".csv":
            return self._load_csv(path)
        raise ValueError(f"不支持的文件类型：{path.suffix}")

    def _load_netcdf(self, path: Path) -> DataAsset:
        ds = xr.open_dataset(path)
        ds = self._standardize_dataset(ds)
        ds = subset_dataset(ds, self.cfg)
        canonical = canonicalize_vars(ds.data_vars, self.cfg.variable_alias)
        return DataAsset(
            path=path,
            source_type="netcdf",
            dataset=ds,
            canonical_vars=canonical,
            metadata=self._dataset_metadata(ds),
        )

    def _load_grib(self, path: Path) -> DataAsset:
        try:
            ds = xr.open_dataset(path, engine="cfgrib")
        except Exception as exc:
            raise RuntimeError("读取 GRIB 需要安装 cfgrib/eccodes，或文件存在多变量索引冲突") from exc
        ds = self._standardize_dataset(ds)
        ds = subset_dataset(ds, self.cfg)
        canonical = canonicalize_vars(ds.data_vars, self.cfg.variable_alias)
        return DataAsset(
            path=path,
            source_type="grib",
            dataset=ds,
            canonical_vars=canonical,
            metadata=self._dataset_metadata(ds),
        )

    def _load_csv(self, path: Path) -> DataAsset:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        df = subset_dataframe(df, self.cfg)
        canonical = canonicalize_vars(df.columns, self.cfg.variable_alias)
        return DataAsset(
            path=path,
            source_type="csv",
            dataframe=df,
            canonical_vars=canonical,
            metadata=self._dataframe_metadata(df),
        )

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        rename_map = {}
        for name in list(ds.dims) + list(ds.coords):
            if is_lat_name(str(name)) and name != "lat":
                rename_map[name] = "lat"
            elif is_lon_name(str(name)) and name != "lon":
                rename_map[name] = "lon"
            elif is_time_name(str(name)) and name != "time":
                rename_map[name] = "time"
        if rename_map:
            ds = ds.rename(rename_map)

        if "lon" in ds.coords:
            lon = ds["lon"]
            try:
                if float(lon.max()) > 180.0:
                    ds = ds.assign_coords(lon=((lon + 180) % 360) - 180).sortby("lon")
            except Exception:
                pass
        return ds

    def _dataset_metadata(self, ds: xr.Dataset) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "variables": list(map(str, ds.data_vars)),
            "coords": list(map(str, ds.coords)),
        }
        if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
            meta["time_start"] = str(pd.to_datetime(ds["time"].values[0]))
            meta["time_end"] = str(pd.to_datetime(ds["time"].values[-1]))
        if "lat" in ds.coords:
            meta["lat_range"] = [float(ds["lat"].min()), float(ds["lat"].max())]
        if "lon" in ds.coords:
            meta["lon_range"] = [float(ds["lon"].min()), float(ds["lon"].max())]
        return meta

    def _dataframe_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
        }
        time_col = find_first_existing(df.columns, ["time", "datetime", "date", "timestamp", "时间"])
        if time_col and len(df) > 0:
            t = pd.to_datetime(df[time_col], errors="coerce")
            meta["time_start"] = str(t.min())
            meta["time_end"] = str(t.max())
        return meta


# -----------------------------
# Agent 2：质量控制
# -----------------------------

class QualityControlAgent:
    def __init__(self, cfg: AgentConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def run(self, assets: List[DataAsset]) -> List[QCRecord]:
        records: List[QCRecord] = []
        for asset in assets:
            if asset.dataset is not None:
                records.extend(self._qc_dataset(asset))
            elif asset.dataframe is not None:
                records.extend(self._qc_dataframe(asset))
        self.logger.info("质控完成：生成 %d 条质控记录", len(records))
        return records

    def _qc_dataset(self, asset: DataAsset) -> List[QCRecord]:
        ds = asset.dataset
        assert ds is not None
        records: List[QCRecord] = []

        if len(ds.data_vars) == 0:
            records.append(QCRecord(str(asset.path), "-", "empty_dataset", "error", "资料中没有可用变量"))
            return records

        for canon, var in asset.canonical_vars.items():
            da = ds[var]
            values = da.values
            total = values.size
            missing = int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.number) else 0
            ratio = missing / max(total, 1)
            records.append(QCRecord(
                str(asset.path), var, "missing", "warning" if ratio > 0.05 else "info",
                f"缺测率 {ratio:.2%}", count=missing, ratio=ratio
            ))

            if canon in self.cfg.physical_ranges and np.issubdtype(values.dtype, np.number):
                lo, hi = self.cfg.physical_ranges[canon]
                bad = np.isfinite(values) & ((values < lo) | (values > hi))
                count = int(bad.sum())
                if count > 0:
                    records.append(QCRecord(
                        str(asset.path), var, "physical_range", "warning",
                        f"超出物理阈值范围 [{lo}, {hi}] 的格点数：{count}", count=count, ratio=count / max(total, 1)
                    ))

            if np.issubdtype(values.dtype, np.number):
                records.extend(self._qc_iqr(asset.path, var, values))
                records.extend(self._qc_temporal_jump_dataset(asset.path, var, da))
        return records

    def _qc_dataframe(self, asset: DataAsset) -> List[QCRecord]:
        df = asset.dataframe
        assert df is not None
        records: List[QCRecord] = []

        if len(df) == 0:
            records.append(QCRecord(str(asset.path), "-", "empty_dataframe", "error", "CSV 过滤后没有记录"))
            return records

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        candidate_vars = set(asset.canonical_vars.values()) | set(
            c for c in numeric_cols if c.lower() in [x.lower() for x in self.cfg.station_value_columns]
        )

        for var in sorted(candidate_vars):
            if var not in df.columns:
                continue
            values = pd.to_numeric(df[var], errors="coerce").to_numpy()
            total = len(values)
            missing = int(np.isnan(values).sum())
            ratio = missing / max(total, 1)
            records.append(QCRecord(
                str(asset.path), var, "missing", "warning" if ratio > 0.05 else "info",
                f"缺测率 {ratio:.2%}", count=missing, ratio=ratio
            ))

            canon = next((k for k, v in asset.canonical_vars.items() if v == var), var)
            if canon in self.cfg.physical_ranges:
                lo, hi = self.cfg.physical_ranges[canon]
                bad = np.isfinite(values) & ((values < lo) | (values > hi))
                count = int(bad.sum())
                if count > 0:
                    records.append(QCRecord(
                        str(asset.path), var, "physical_range", "warning",
                        f"超出物理阈值范围 [{lo}, {hi}] 的记录数：{count}", count=count, ratio=count / max(total, 1)
                    ))

            records.extend(self._qc_iqr(asset.path, var, values))
            records.extend(self._qc_temporal_jump_dataframe(asset.path, var, df))
            records.extend(self._qc_spatial_consistency(asset.path, var, df))

        return records

    def _qc_iqr(self, path: Path, var: str, values: np.ndarray) -> List[QCRecord]:
        finite = values[np.isfinite(values)]
        if finite.size < 20:
            return []
        q1, q3 = np.nanpercentile(finite, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return []
        low = q1 - 3.0 * iqr
        high = q3 + 3.0 * iqr
        outlier = np.isfinite(values) & ((values < low) | (values > high))
        count = int(outlier.sum())
        ratio = count / max(values.size, 1)
        if ratio > 0.01:
            return [QCRecord(
                str(path), var, "iqr_outlier", "warning",
                f"IQR 异常值比例 {ratio:.2%}，阈值约 [{low:.3f}, {high:.3f}]",
                count=count, ratio=ratio
            )]
        return []

    def _qc_temporal_jump_dataset(self, path: Path, var: str, da: xr.DataArray) -> List[QCRecord]:
        if "time" not in da.dims or da.sizes.get("time", 0) < 3:
            return []
        try:
            spatial_dims = [d for d in da.dims if d != "time"]
            series = da.mean(dim=spatial_dims, skipna=True).values if spatial_dims else da.values
            diff = np.diff(series.astype(float))
            finite = diff[np.isfinite(diff)]
            if finite.size < 5:
                return []
            threshold = np.nanmedian(np.abs(finite)) + 6 * np.nanstd(finite)
            jumps = np.abs(diff) > threshold if threshold > 0 else np.zeros_like(diff, dtype=bool)
            count = int(jumps.sum())
            if count > 0:
                return [QCRecord(str(path), var, "temporal_jump", "warning", f"时间序列均值存在疑似跳变：{count} 次", count=count)]
        except Exception:
            return []
        return []

    def _qc_temporal_jump_dataframe(self, path: Path, var: str, df: pd.DataFrame) -> List[QCRecord]:
        time_col = find_first_existing(df.columns, ["time", "datetime", "date", "timestamp", "时间"])
        station_col = find_first_existing(df.columns, ["station", "station_id", "id", "站号"])
        if not time_col:
            return []
        try:
            tmp = df[[time_col, var] + ([station_col] if station_col else [])].copy()
            tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
            tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
            tmp = tmp.dropna(subset=[time_col, var]).sort_values(time_col)
            if station_col:
                grouped = tmp.groupby(station_col)
            else:
                grouped = [("all", tmp)]
            jump_count = 0
            for _, g in grouped:
                if len(g) < 5:
                    continue
                diff = g[var].diff().dropna().to_numpy()
                if diff.size < 4:
                    continue
                threshold = np.nanmedian(np.abs(diff)) + 6 * np.nanstd(diff)
                if threshold > 0:
                    jump_count += int((np.abs(diff) > threshold).sum())
            if jump_count > 0:
                return [QCRecord(str(path), var, "temporal_jump", "warning", f"站点时间序列存在疑似跳变：{jump_count} 次", count=jump_count)]
        except Exception:
            return []
        return []

    def _qc_spatial_consistency(self, path: Path, var: str, df: pd.DataFrame) -> List[QCRecord]:
        if cKDTree is None:
            return []
        lat_col = find_first_existing(df.columns, ["lat", "latitude", "纬度"])
        lon_col = find_first_existing(df.columns, ["lon", "longitude", "经度"])
        time_col = find_first_existing(df.columns, ["time", "datetime", "date", "timestamp", "时间"])
        if not lat_col or not lon_col:
            return []
        try:
            cols = [lat_col, lon_col, var] + ([time_col] if time_col else [])
            tmp = df[cols].copy()
            tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
            tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
            tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
            tmp = tmp.dropna(subset=[lat_col, lon_col, var])
            if len(tmp) < 10:
                return []

            if time_col:
                tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
                groups = tmp.groupby(time_col)
            else:
                groups = [("all", tmp)]

            abnormal = 0
            total = 0
            for _, g in groups:
                if len(g) < 10:
                    continue
                xy = g[[lon_col, lat_col]].to_numpy(dtype=float)
                z = g[var].to_numpy(dtype=float)
                tree = cKDTree(xy)
                k = min(6, len(g))
                _, idx = tree.query(xy, k=k)
                if idx.ndim == 1:
                    continue
                neigh = idx[:, 1:]
                neigh_mean = np.nanmean(z[neigh], axis=1)
                residual = z - neigh_mean
                mad = np.nanmedian(np.abs(residual - np.nanmedian(residual)))
                threshold = 6 * 1.4826 * mad
                if threshold > 0:
                    abnormal += int((np.abs(residual) > threshold).sum())
                    total += len(g)
            if total > 0 and abnormal / total > 0.01:
                return [QCRecord(str(path), var, "spatial_consistency", "warning", f"空间一致性疑似异常比例 {abnormal / total:.2%}", count=abnormal, ratio=abnormal / total)]
        except Exception:
            return []
        return []


# -----------------------------
# Agent 3：诊断分析
# -----------------------------

class DiagnosticAgent:
    def __init__(self, cfg: AgentConfig, logger: logging.Logger, out_dirs: Dict[str, Path]):
        self.cfg = cfg
        self.logger = logger
        self.out_dirs = out_dirs

    def run(self, assets: List[DataAsset]) -> List[DiagnosticResult]:
        event = self._infer_event_type(assets)
        self.logger.info("诊断类型：%s", event)
        results: List[DiagnosticResult] = []

        results.append(self._inventory_summary(assets))
        if event == "heavy_rain":
            results.extend(self._diagnose_heavy_rain(assets))
        elif event == "heatwave":
            results.extend(self._diagnose_heatwave(assets))
        elif event == "cold_wave":
            results.extend(self._diagnose_cold_wave(assets))
        elif event == "typhoon":
            results.extend(self._diagnose_typhoon(assets))
        else:
            results.extend(self._diagnose_general(assets))

        return results

    def _infer_event_type(self, assets: List[DataAsset]) -> str:
        if self.cfg.event_type != "auto":
            return self.cfg.event_type
        available = set()
        for a in assets:
            available |= set(a.canonical_vars.keys())
        if "precip" in available or "reflectivity" in available:
            return "heavy_rain"
        if "temperature" in available:
            return "heatwave"
        return "general"

    def _inventory_summary(self, assets: List[DataAsset]) -> DiagnosticResult:
        rows = []
        for a in assets:
            rows.append({
                "file": a.path.name,
                "type": a.source_type,
                "canonical_vars": ", ".join(f"{k}:{v}" for k, v in a.canonical_vars.items()),
                "metadata": json.dumps(a.metadata, ensure_ascii=False),
            })
        table = pd.DataFrame(rows)
        table_path = self.out_dirs["tables"] / "inventory.csv"
        table.to_csv(table_path, index=False, encoding="utf-8-sig")
        return DiagnosticResult(
            name="资料清单",
            text=f"共接入 {len(assets)} 个资料文件，资料清单已保存至 {table_path.name}。",
            tables={"inventory": table},
            metrics={"asset_count": len(assets)},
        )

    def _diagnose_heavy_rain(self, assets: List[DataAsset]) -> List[DiagnosticResult]:
        results: List[DiagnosticResult] = []
        precip_assets = [a for a in assets if "precip" in a.canonical_vars]
        refl_assets = [a for a in assets if "reflectivity" in a.canonical_vars]

        for asset in precip_assets:
            results.append(self._precip_summary(asset))

        for asset in refl_assets:
            results.append(self._reflectivity_summary(asset))

        wind_assets = [a for a in assets if "u_wind" in a.canonical_vars and "v_wind" in a.canonical_vars]
        for asset in wind_assets:
            results.append(self._wind_summary(asset))

        if not precip_assets and not refl_assets:
            results.extend(self._diagnose_general(assets))
        return results

    def _diagnose_heatwave(self, assets: List[DataAsset]) -> List[DiagnosticResult]:
        results = []
        for asset in [a for a in assets if "temperature" in a.canonical_vars]:
            results.append(self._temperature_summary(asset, mode="heatwave"))
        if not results:
            results.extend(self._diagnose_general(assets))
        return results

    def _diagnose_cold_wave(self, assets: List[DataAsset]) -> List[DiagnosticResult]:
        results = []
        for asset in [a for a in assets if "temperature" in a.canonical_vars]:
            results.append(self._temperature_summary(asset, mode="cold_wave"))
        if not results:
            results.extend(self._diagnose_general(assets))
        return results

    def _diagnose_typhoon(self, assets: List[DataAsset]) -> List[DiagnosticResult]:
        results = []
        for asset in [a for a in assets if "mslp" in a.canonical_vars]:
            results.append(self._pressure_summary(asset))
        for asset in [a for a in assets if "u_wind" in a.canonical_vars and "v_wind" in a.canonical_vars]:
            results.append(self._wind_summary(asset))
        for asset in [a for a in assets if "precip" in a.canonical_vars]:
            results.append(self._precip_summary(asset))
        if not results:
            results.extend(self._diagnose_general(assets))
        return results

    def _diagnose_general(self, assets: List[DataAsset]) -> List[DiagnosticResult]:
        results: List[DiagnosticResult] = []
        for asset in assets:
            if asset.dataset is not None:
                for canon, var in asset.canonical_vars.items():
                    results.append(self._generic_dataset_variable_summary(asset, canon, var))
            elif asset.dataframe is not None:
                for canon, var in asset.canonical_vars.items():
                    results.append(self._generic_dataframe_variable_summary(asset, canon, var))
        return results

    def _precip_summary(self, asset: DataAsset) -> DiagnosticResult:
        var = asset.canonical_vars["precip"]
        figures: List[Path] = []
        metrics: Dict[str, Any] = {}

        if asset.dataset is not None:
            da = asset.dataset[var]
            acc = da.sum(dim="time", skipna=True) if "time" in da.dims else da
            metrics = summarize_array(acc.values)
            fig_path = self._plot_grid_or_series(acc, f"累计降水 - {asset.path.name}", f"acc_precip_{safe_name(asset.path.stem)}.png")
            if fig_path:
                figures.append(fig_path)
            text = (
                f"{asset.path.name} 的累计降水最大值为 {metrics.get('max', np.nan):.2f}，"
                f"区域平均值为 {metrics.get('mean', np.nan):.2f}。"
            )
        else:
            df = asset.dataframe
            assert df is not None
            table, fig_path = self._station_aggregate(df, var, agg="sum", title=f"站点累计降水 - {asset.path.name}")
            if fig_path:
                figures.append(fig_path)
            metrics = table[var].pipe(lambda s: summarize_array(s.to_numpy())) if len(table) else {}
            text = (
                f"{asset.path.name} 的站点累计降水最大值为 {metrics.get('max', np.nan):.2f}，"
                f"站点平均值为 {metrics.get('mean', np.nan):.2f}。"
            )
            table_path = self.out_dirs["tables"] / f"station_acc_precip_{safe_name(asset.path.stem)}.csv"
            table.to_csv(table_path, index=False, encoding="utf-8-sig")

        return DiagnosticResult("强降水诊断", text, figures=figures, metrics=metrics)

    def _reflectivity_summary(self, asset: DataAsset) -> DiagnosticResult:
        var = asset.canonical_vars["reflectivity"]
        figures: List[Path] = []
        if asset.dataset is not None:
            da = asset.dataset[var]
            vmax = da.max(dim="time", skipna=True) if "time" in da.dims else da
            metrics = summarize_array(vmax.values)
            fig_path = self._plot_grid_or_series(vmax, f"最大雷达回波 - {asset.path.name}", f"max_reflectivity_{safe_name(asset.path.stem)}.png")
            if fig_path:
                figures.append(fig_path)
            text = f"{asset.path.name} 的最大回波强度为 {metrics.get('max', np.nan):.2f} dBZ。"
        else:
            metrics = {}
            text = f"{asset.path.name} 包含雷达回波字段，但 CSV 回波诊断暂采用统计摘要。"
        return DiagnosticResult("雷达回波诊断", text, figures=figures, metrics=metrics)

    def _wind_summary(self, asset: DataAsset) -> DiagnosticResult:
        u_var = asset.canonical_vars["u_wind"]
        v_var = asset.canonical_vars["v_wind"]
        figures: List[Path] = []

        if asset.dataset is not None:
            ds = asset.dataset
            wspd = np.sqrt(ds[u_var] ** 2 + ds[v_var] ** 2)
            if "time" in wspd.dims:
                wspd2 = wspd.max(dim="time", skipna=True)
            else:
                wspd2 = wspd
            metrics = summarize_array(wspd2.values)
            fig_path = self._plot_grid_or_series(wspd2, f"最大风速 - {asset.path.name}", f"max_wind_{safe_name(asset.path.stem)}.png")
            if fig_path:
                figures.append(fig_path)
            text = f"{asset.path.name} 的最大风速为 {metrics.get('max', np.nan):.2f}。"
        else:
            df = asset.dataframe
            assert df is not None
            tmp = df.copy()
            tmp["wind_speed"] = np.sqrt(pd.to_numeric(tmp[u_var], errors="coerce") ** 2 + pd.to_numeric(tmp[v_var], errors="coerce") ** 2)
            metrics = summarize_array(tmp["wind_speed"].to_numpy())
            fig_path = self._station_scatter(tmp, "wind_speed", f"站点风速 - {asset.path.name}", f"station_wind_{safe_name(asset.path.stem)}.png")
            if fig_path:
                figures.append(fig_path)
            text = f"{asset.path.name} 的站点最大风速为 {metrics.get('max', np.nan):.2f}。"

        return DiagnosticResult("风场诊断", text, figures=figures, metrics=metrics)

    def _temperature_summary(self, asset: DataAsset, mode: str) -> DiagnosticResult:
        var = asset.canonical_vars["temperature"]
        figures: List[Path] = []

        if asset.dataset is not None:
            da = asset.dataset[var]
            target = da.max(dim="time", skipna=True) if mode == "heatwave" and "time" in da.dims else da.min(dim="time", skipna=True) if "time" in da.dims else da
            metrics = summarize_array(target.values)
            fig_name = f"temperature_{mode}_{safe_name(asset.path.stem)}.png"
            fig_path = self._plot_grid_or_series(target, f"温度诊断 - {asset.path.name}", fig_name)
            if fig_path:
                figures.append(fig_path)
        else:
            df = asset.dataframe
            assert df is not None
            metrics = summarize_array(pd.to_numeric(df[var], errors="coerce").to_numpy())
            fig_path = self._station_scatter(df, var, f"站点温度 - {asset.path.name}", f"station_temperature_{safe_name(asset.path.stem)}.png")
            if fig_path:
                figures.append(fig_path)

        if mode == "heatwave":
            text = f"{asset.path.name} 的最高温度为 {metrics.get('max', np.nan):.2f}。"
        else:
            text = f"{asset.path.name} 的最低温度为 {metrics.get('min', np.nan):.2f}。"
        return DiagnosticResult("温度诊断", text, figures=figures, metrics=metrics)

    def _pressure_summary(self, asset: DataAsset) -> DiagnosticResult:
        var = asset.canonical_vars["mslp"]
        figures: List[Path] = []
        if asset.dataset is not None:
            da = asset.dataset[var]
            target = da.min(dim="time", skipna=True) if "time" in da.dims else da
            metrics = summarize_array(target.values)
            fig_path = self._plot_grid_or_series(target, f"最低海平面气压 - {asset.path.name}", f"min_mslp_{safe_name(asset.path.stem)}.png")
            if fig_path:
                figures.append(fig_path)
        else:
            df = asset.dataframe
            assert df is not None
            values = pd.to_numeric(df[var], errors="coerce")
            metrics = summarize_array(values.to_numpy())
        text = f"{asset.path.name} 的最低海平面气压为 {metrics.get('min', np.nan):.2f}。"
        return DiagnosticResult("台风气压诊断", text, figures=figures, metrics=metrics)

    def _generic_dataset_variable_summary(self, asset: DataAsset, canon: str, var: str) -> DiagnosticResult:
        assert asset.dataset is not None
        da = asset.dataset[var]
        metrics = summarize_array(da.values)
        fig_path = self._plot_grid_or_series(da.isel(time=0) if "time" in da.dims else da, f"{canon} - {asset.path.name}", f"generic_{canon}_{safe_name(asset.path.stem)}.png")
        figures = [fig_path] if fig_path else []
        text = f"{asset.path.name} 中 {var} 的统计范围：min={metrics.get('min', np.nan):.2f}, max={metrics.get('max', np.nan):.2f}, mean={metrics.get('mean', np.nan):.2f}。"
        return DiagnosticResult(f"通用变量诊断：{canon}", text, figures=figures, metrics=metrics)

    def _generic_dataframe_variable_summary(self, asset: DataAsset, canon: str, var: str) -> DiagnosticResult:
        assert asset.dataframe is not None
        values = pd.to_numeric(asset.dataframe[var], errors="coerce").to_numpy()
        metrics = summarize_array(values)
        fig_path = self._station_scatter(asset.dataframe, var, f"{canon} - {asset.path.name}", f"generic_station_{canon}_{safe_name(asset.path.stem)}.png")
        figures = [fig_path] if fig_path else []
        text = f"{asset.path.name} 中 {var} 的统计范围：min={metrics.get('min', np.nan):.2f}, max={metrics.get('max', np.nan):.2f}, mean={metrics.get('mean', np.nan):.2f}。"
        return DiagnosticResult(f"通用变量诊断：{canon}", text, figures=figures, metrics=metrics)

    def _plot_grid_or_series(self, da: xr.DataArray, title: str, filename: str) -> Optional[Path]:
        fig_path = self.out_dirs["figures"] / filename
        try:
            da2 = da.squeeze(drop=True)
            plt.figure(figsize=(8, 5))
            if "lat" in da2.dims and "lon" in da2.dims:
                da2.plot(cmap="viridis")
                plt.title(title)
            elif da2.ndim == 1:
                values = da2.values.astype(float)
                plt.plot(np.arange(len(values)), values)
                plt.title(title)
                plt.xlabel("index")
                plt.ylabel(str(da2.name or "value"))
            elif da2.ndim == 2:
                plt.imshow(da2.values.astype(float), origin="lower", aspect="auto")
                plt.colorbar(label=str(da2.name or "value"))
                plt.title(title)
            else:
                values = da2.values.astype(float).ravel()
                plt.plot(np.arange(len(values)), values)
                plt.title(title)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
            plt.close()
            return fig_path
        except Exception as exc:
            self.logger.warning("绘图失败：%s | %s", title, exc)
            plt.close()
            return None

    def _station_aggregate(self, df: pd.DataFrame, var: str, agg: str, title: str) -> Tuple[pd.DataFrame, Optional[Path]]:
        lat_col = find_first_existing(df.columns, ["lat", "latitude", "纬度"])
        lon_col = find_first_existing(df.columns, ["lon", "longitude", "经度"])
        station_col = find_first_existing(df.columns, ["station", "station_id", "id", "站号"])
        if station_col and lat_col and lon_col:
            grouped = df.groupby(station_col).agg({lat_col: "first", lon_col: "first", var: agg}).reset_index()
        elif lat_col and lon_col:
            grouped = df.groupby([lat_col, lon_col]).agg({var: agg}).reset_index()
        else:
            grouped = pd.DataFrame({var: [pd.to_numeric(df[var], errors="coerce").agg(agg)]})
        fig_path = self._station_scatter(grouped, var, title, f"{safe_name(title)}.png")
        return grouped, fig_path

    def _station_scatter(self, df: pd.DataFrame, var: str, title: str, filename: str) -> Optional[Path]:
        lat_col = find_first_existing(df.columns, ["lat", "latitude", "纬度"])
        lon_col = find_first_existing(df.columns, ["lon", "longitude", "经度"])
        fig_path = self.out_dirs["figures"] / filename
        try:
            values = pd.to_numeric(df[var], errors="coerce")
            plt.figure(figsize=(8, 5))
            if lat_col and lon_col:
                plt.scatter(pd.to_numeric(df[lon_col], errors="coerce"), pd.to_numeric(df[lat_col], errors="coerce"), c=values, s=28)
                plt.colorbar(label=var)
                plt.xlabel("lon")
                plt.ylabel("lat")
            else:
                plt.plot(np.arange(len(values)), values)
                plt.xlabel("index")
                plt.ylabel(var)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
            plt.close()
            return fig_path
        except Exception as exc:
            self.logger.warning("站点绘图失败：%s | %s", title, exc)
            plt.close()
            return None


# -----------------------------
# Agent 4：报告生成
# -----------------------------

class ReportAgent:
    def __init__(self, cfg: AgentConfig, logger: logging.Logger, out_dirs: Dict[str, Path]):
        self.cfg = cfg
        self.logger = logger
        self.out_dirs = out_dirs

    def write(self, assets: List[DataAsset], qc_records: List[QCRecord], diagnostics: List[DiagnosticResult]) -> Path:
        report_path = self.cfg.output_dir / "case_report.md"
        qc_table = pd.DataFrame([asdict(r) for r in qc_records])
        qc_path = self.out_dirs["tables"] / "qc_records.csv"
        qc_table.to_csv(qc_path, index=False, encoding="utf-8-sig")

        manifest_path = self.cfg.output_dir / "manifest.json"
        manifest = {
            "case_name": self.cfg.case_name,
            "event_type": self.cfg.event_type,
            "start_time": str(self.cfg.start_time) if self.cfg.start_time is not None else None,
            "end_time": str(self.cfg.end_time) if self.cfg.end_time is not None else None,
            "region": self.cfg.region,
            "asset_count": len(assets),
            "qc_record_count": len(qc_records),
            "diagnostic_count": len(diagnostics),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        lines: List[str] = []
        lines.append(f"# {self.cfg.case_name}\n")
        lines.append("## 1. 个例基本信息\n")
        lines.append(f"- 个例类型：{self.cfg.event_type}\n")
        lines.append(f"- 分析时段：{self.cfg.start_time or '未指定'} 至 {self.cfg.end_time or '未指定'}\n")
        lines.append(f"- 分析区域：{self.cfg.region or '未指定'}\n")
        lines.append(f"- 资料数量：{len(assets)}\n")
        lines.append(f"- 生成时间：{manifest['generated_at']}\n")

        lines.append("\n## 2. 资料接入概况\n")
        for asset in assets:
            lines.append(f"### {asset.path.name}\n")
            lines.append(f"- 类型：{asset.source_type}\n")
            lines.append(f"- 识别变量：{', '.join(f'{k}={v}' for k, v in asset.canonical_vars.items()) or '未识别'}\n")
            lines.append(f"- 元数据：`{json.dumps(asset.metadata, ensure_ascii=False)}`\n")

        lines.append("\n## 3. 质量控制结果\n")
        if qc_table.empty:
            lines.append("未生成质控记录。\n")
        else:
            severity_counts = qc_table["severity"].value_counts().to_dict()
            lines.append(f"- 质控记录总数：{len(qc_table)}\n")
            lines.append(f"- 分级统计：{severity_counts}\n")
            warn = qc_table[qc_table["severity"].isin(["warning", "error"])]
            if len(warn) > 0:
                lines.append("\n主要告警：\n")
                for _, row in warn.head(30).iterrows():
                    lines.append(f"- `{Path(str(row['asset'])).name}` / `{row['variable']}` / {row['check_name']}：{row['message']}\n")
            lines.append(f"\n完整质控表：`tables/{qc_path.name}`\n")

        lines.append("\n## 4. 诊断结论\n")
        for i, d in enumerate(diagnostics, start=1):
            lines.append(f"### 4.{i} {d.name}\n")
            lines.append(d.text + "\n")
            if d.metrics:
                lines.append(f"- 指标摘要：`{json.dumps(d.metrics, ensure_ascii=False)}`\n")
            for fig in d.figures:
                rel = fig.relative_to(self.cfg.output_dir)
                lines.append(f"\n![{d.name}]({rel.as_posix()})\n")

        lines.append("\n## 5. 自动研判摘要\n")
        lines.append(self._generate_interpretation(qc_records, diagnostics))

        lines.append("\n## 6. 人工复核建议\n")
        lines.append("- 检查资料时间范围是否覆盖天气过程发生、发展和结束阶段。\n")
        lines.append("- 对质控告警中的极端站点或格点，结合雷达、卫星、邻近站点进行二次确认。\n")
        lines.append("- 若用于论文或业务发布，应补充环流背景、水汽输送、动力抬升和地形影响等人工解释。\n")
        lines.append("- 当前代码提供基础诊断框架，业务阈值和区域规则应按本地标准进一步配置。\n")

        report_path.write_text("".join(lines), encoding="utf-8")
        self.logger.info("报告已生成：%s", report_path)
        return report_path

    def _generate_interpretation(self, qc_records: List[QCRecord], diagnostics: List[DiagnosticResult]) -> str:
        warnings = [r for r in qc_records if r.severity in {"warning", "error"}]
        lines = []
        if warnings:
            lines.append(f"本次个例资料存在 {len(warnings)} 条需要关注的质控告警，建议优先核查缺测率较高、物理阈值越界和空间一致性异常的记录。\n")
        else:
            lines.append("本次个例资料未发现显著质控告警，可进入后续诊断分析。\n")

        for d in diagnostics:
            if "强降水" in d.name and d.metrics:
                maxv = d.metrics.get("max")
                meanv = d.metrics.get("mean")
                if maxv is not None:
                    lines.append(f"降水诊断显示过程最大累计降水约为 {maxv:.2f}，区域或站点平均约为 {meanv:.2f}，应重点关注高值中心及其与低空急流、地形抬升和雷达回波的对应关系。\n")
            if "风场" in d.name and d.metrics:
                maxv = d.metrics.get("max")
                if maxv is not None:
                    lines.append(f"风场诊断显示最大风速约为 {maxv:.2f}，可进一步结合水汽通量和辐合区判断触发与维持机制。\n")
            if "温度" in d.name and d.metrics:
                lines.append("温度诊断可用于识别高温或降温中心，建议结合边界层高度、云量和冷暖平流进一步判断成因。\n")
        return "".join(lines)


# -----------------------------
# 总控 Orchestrator
# -----------------------------

class MeteoAgentOrchestrator:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.out_dirs = ensure_output_dirs(cfg.output_dir)
        self.logger = setup_logger(cfg.output_dir)
        self.ingestion = DataIngestionAgent(cfg, self.logger)
        self.qc = QualityControlAgent(cfg, self.logger)
        self.diagnostic = DiagnosticAgent(cfg, self.logger, self.out_dirs)
        self.report = ReportAgent(cfg, self.logger, self.out_dirs)

    def run(self) -> Path:
        self.logger.info("启动气象多源资料质控与个例诊断 Agent")
        self.logger.info("输入目录：%s", self.cfg.input_dir)
        self.logger.info("输出目录：%s", self.cfg.output_dir)

        assets = self.ingestion.load_all()
        if not assets:
            raise RuntimeError("没有成功接入任何资料。请检查输入目录和文件格式。")

        qc_records = self.qc.run(assets)
        diagnostics = self.diagnostic.run(assets)
        report_path = self.report.write(assets, qc_records, diagnostics)
        self.logger.info("全部流程完成")
        return report_path


# -----------------------------
# 演示数据生成
# -----------------------------

def make_sample_data(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)

    times = pd.date_range("2024-07-20 00:00", periods=24, freq="h")
    lat = np.linspace(30.0, 35.0, 51)
    lon = np.linspace(110.0, 116.0, 61)
    lon2d, lat2d = np.meshgrid(lon, lat)

    rng = np.random.default_rng(42)
    precip = []
    u10 = []
    v10 = []
    temp = []
    for i, _ in enumerate(times):
        cx = 112.0 + 0.08 * i
        cy = 32.0 + 0.03 * i
        core = np.exp(-((lon2d - cx) ** 2 / 1.2 + (lat2d - cy) ** 2 / 0.8))
        rain = 8 * core + rng.gamma(0.6, 0.7, size=core.shape)
        precip.append(rain)
        u10.append(6 + 2 * np.sin((lat2d - 30) / 5 * np.pi) + rng.normal(0, 0.5, size=core.shape))
        v10.append(3 + 2 * np.cos((lon2d - 110) / 6 * np.pi) + rng.normal(0, 0.5, size=core.shape))
        temp.append(28 + 2 * np.sin(i / 24 * 2 * np.pi) - 0.8 * core + rng.normal(0, 0.3, size=core.shape))

    ds = xr.Dataset(
        {
            "precip": (("time", "lat", "lon"), np.asarray(precip)),
            "u10": (("time", "lat", "lon"), np.asarray(u10)),
            "v10": (("time", "lat", "lon"), np.asarray(v10)),
            "t2m": (("time", "lat", "lon"), np.asarray(temp)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"description": "sample gridded weather data"},
    )
    ds.to_netcdf(input_dir / "sample_grid.nc")

    n_station = 80
    st_lats = rng.uniform(30.1, 34.9, n_station)
    st_lons = rng.uniform(110.1, 115.9, n_station)
    rows = []
    for sid in range(n_station):
        for t in times:
            i = int((t - times[0]) / pd.Timedelta(hours=1))
            cx = 112.0 + 0.08 * i
            cy = 32.0 + 0.03 * i
            core = math.exp(-((st_lons[sid] - cx) ** 2 / 1.2 + (st_lats[sid] - cy) ** 2 / 0.8))
            rain = max(0, 8 * core + rng.gamma(0.6, 0.7) - 0.5)
            rows.append({
                "station_id": f"S{sid:03d}",
                "time": t,
                "lat": st_lats[sid],
                "lon": st_lons[sid],
                "precip": rain,
                "temperature": 28 + rng.normal(0, 1),
            })
    df = pd.DataFrame(rows)
    df.loc[df.sample(frac=0.01, random_state=1).index, "precip"] = np.nan
    df.to_csv(input_dir / "sample_station.csv", index=False, encoding="utf-8-sig")


# -----------------------------
# CLI
# -----------------------------

def parse_region(region_text: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not region_text:
        return None
    parts = [float(x.strip()) for x in region_text.split(",")]
    if len(parts) != 4:
        raise ValueError("region 格式应为 lon_min,lon_max,lat_min,lat_max")
    return parts[0], parts[1], parts[2], parts[3]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="气象多源资料质控与个例诊断 Agent")
    parser.add_argument("--input", required=True, help="输入资料目录")
    parser.add_argument("--output", default="./meteo_agent_outputs", help="输出目录")
    parser.add_argument("--case-name", default="气象多源资料质控与个例诊断", help="个例名称")
    parser.add_argument("--event", default="auto", choices=["auto", "heavy_rain", "typhoon", "heatwave", "cold_wave", "convection", "general"], help="个例类型")
    parser.add_argument("--start", default=None, help="开始时间，例如 2024-07-20 00:00")
    parser.add_argument("--end", default=None, help="结束时间，例如 2024-07-21 00:00")
    parser.add_argument("--region", default=None, help="区域范围：lon_min,lon_max,lat_min,lat_max")
    parser.add_argument("--max-files", type=int, default=200, help="最多接入文件数")
    parser.add_argument("--make-sample", action="store_true", help="生成一套演示资料到 input 目录后运行")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if args.make_sample:
        make_sample_data(input_dir)

    cfg = AgentConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        case_name=args.case_name,
        event_type=args.event,
        start_time=parse_timestamp(args.start),
        end_time=parse_timestamp(args.end),
        region=parse_region(args.region),
        max_files=args.max_files,
    )

    orchestrator = MeteoAgentOrchestrator(cfg)
    report_path = orchestrator.run()
    print(f"\n完成。报告路径：{report_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
