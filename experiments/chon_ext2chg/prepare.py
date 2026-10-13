"""Build the standard CHON cache without a whole-dataset PAW mean.

The benchmark fits its occupancy baseline on training atoms only. POTCAR
profiles are provenance, not fitted target statistics.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from poraque.data import build_field_cache, discover_records, resolve_source
from poraque.data.cache import build_paw_profiles
from poraque.ml.config import TrainingConfig
from poraque_train import cache_tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_ext2chg_chon.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = TrainingConfig.from_yaml(args.config)
    data = config.data
    cache = str(Path(data.cache) / cache_tag(data))
    build_field_cache(data.paths(), cache, resolution=data.resolution,
                      potcar_dir=data.potcar_dir, strict_potcar=True,
                      spin=data.spin, storage=data.storage, limit=args.limit,
                      log=lambda s: print(s, flush=True))
    sources = [resolve_source(p, potcar_dir=data.potcar_dir) for p in data.paths()]
    build_paw_profiles(discover_records(sources, required=("CHGCAR",)),
                       cache, log=print)
    print(cache, flush=True)


if __name__ == "__main__":
    main()
