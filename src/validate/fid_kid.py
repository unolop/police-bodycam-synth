"""FID/KID computation using clean-fid."""

from pathlib import Path


def compute_fid_kid(generated_dir: Path, real_dir: Path | None) -> dict:
    """Compute FID and KID between generated and real image directories."""
    if real_dir is None or not Path(real_dir).exists():
        return {"fid": None, "kid": None, "note": "No reference dataset provided"}

    from cleanfid import fid

    fid_score = fid.compute_fid(str(generated_dir), str(real_dir))
    kid_score = fid.compute_kid(str(generated_dir), str(real_dir))

    return {
        "fid": float(fid_score),
        "kid": float(kid_score),
    }
