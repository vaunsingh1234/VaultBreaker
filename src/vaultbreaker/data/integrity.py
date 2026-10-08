import os
from pathlib import Path
from typing import Union
import pandas as pd
import cv2
import soundfile as sf
import imageio.v3 as iio

def assert_no_leakage(manifest: Union[pd.DataFrame, str, Path]) -> bool:
    """
    Asserts that no source_id appears in more than one split partition.
    Throws AssertionError if leakage is found.
    """
    if isinstance(manifest, (str, Path)):
        df = pd.read_csv(manifest)
    else:
        df = manifest.copy()

    splits = df["split"].unique()
    source_sets = {}
    for s in splits:
        source_sets[s] = set(df[df["split"] == s]["source_id"])

    # Check all pairs
    split_list = list(splits)
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            s1, s2 = split_list[i], split_list[j]
            overlap = source_sets[s1].intersection(source_sets[s2])
            if overlap:
                raise AssertionError(
                    f"CRITICAL LEAKAGE DETECTED: {len(overlap)} sources overlap between {s1} and {s2}! "
                    f"Sample overlapping source IDs: {list(overlap)[:5]}"
                )
    return True

def assert_reencoding_identity(cover_path: Union[str, Path], stego_path: Union[str, Path]) -> bool:
    """
    Asserts that cover and stego share identical container and encoding parameters
    (resolution, bit depth, channel count, sample rate) to ensure the detector cannot
    learn file container/format shortcuts.
    """
    c_p = Path(cover_path)
    s_p = Path(stego_path)
    
    assert c_p.suffix == s_p.suffix, f"Suffix mismatch: {c_p.suffix} vs {s_p.suffix}"
    
    ext = c_p.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        img_c = cv2.imread(str(c_p), cv2.IMREAD_UNCHANGED)
        img_s = cv2.imread(str(s_p), cv2.IMREAD_UNCHANGED)
        assert img_c.shape == img_s.shape, f"Shape mismatch: {img_c.shape} vs {img_s.shape}"
        assert img_c.dtype == img_s.dtype, f"Dtype mismatch: {img_c.dtype} vs {img_s.dtype}"
    elif ext in [".wav"]:
        info_c = sf.info(str(c_p))
        info_s = sf.info(str(s_p))
        assert info_c.samplerate == info_s.samplerate, "Sample rate mismatch"
        assert info_c.channels == info_s.channels, "Channel mismatch"
        assert info_c.subtype == info_s.subtype, "Subtype PCM mismatch"
    elif ext in [".mkv", ".avi", ".mp4"]:
        props_c = iio.improps(str(c_p))
        props_s = iio.improps(str(s_p))
        assert props_c.shape == props_s.shape, f"Video shape mismatch: {props_c.shape} vs {props_s.shape}"
        assert props_c.dtype == props_s.dtype, f"Video dtype mismatch: {props_c.dtype} vs {props_s.dtype}"
    return True
