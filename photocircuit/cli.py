"""Command-line interface: photocircuit <image...> -> CircuiTikZ code.

Examples:
    python -m photocircuit photo.png
    python -m photocircuit --debug-dir out photo.png     # save intermediates
    python -m photocircuit --no-model photo.png          # skip classification
"""

import argparse
import sys
from pathlib import Path

import cv2


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="photocircuit",
        description="Recognize a hand-drawn circuit photo and emit CircuiTikZ code.")
    p.add_argument("images", nargs="+", help="input photo(s), any format cv2.imread reads")
    p.add_argument("--model", type=Path, default=None,
                   help="path to the .tflite classifier (default: the one from the Android app)")
    p.add_argument("--no-model", action="store_true",
                   help="skip classification (elements become unknown/dangling)")
    p.add_argument("--portrait", action="store_true",
                   help="treat input as portrait (matches the app's camera toggle)")
    p.add_argument("--debug-dir", type=Path, default=None,
                   help="save intermediate images and per-element crops here")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="write TikZ code to this file instead of stdout")
    # parse_intermixed_args allows options after the positional images list,
    # e.g. `photocircuit photo.png -o out.tex`.
    return p.parse_intermixed_args(argv)


def _dump_debug(debug_dir: Path, stem: str, result) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    proc = result.processing
    cv2.imwrite(str(debug_dir / f"{stem}_threshold.png"), proc.thresholded_image)
    cv2.imwrite(str(debug_dir / f"{stem}_thinned.png"), proc.thinned_image)

    annotated = cv2.cvtColor(proc.thresholded_image, cv2.COLOR_GRAY2BGR)
    for e in result.elements:
        r = e.rect
        cv2.rectangle(annotated, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 255, 255), 3)
        cv2.putText(annotated, e.best_guess, (r.x, max(r.y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(str(debug_dir / f"{stem}_rects.png"), annotated)

    for i, e in enumerate(result.elements):
        r = e.rect
        crop = proc.thresholded_image[r.y:r.y + r.h, r.x:r.x + r.w]
        if crop.size:
            cv2.imwrite(str(debug_dir / f"{stem}_elem{i}_{e.best_guess}.png"), crop)


def main(argv=None) -> int:
    args = _parse_args(argv)

    from .pipeline import PhotocircuitPipeline, run

    exit_code = 0
    outputs = []
    for path in args.images:
        try:
            if args.no_model:
                img = cv2.imread(str(path))
                if img is None:
                    raise FileNotFoundError(path)
                result = PhotocircuitPipeline().run(img, landscape=not args.portrait)
            else:
                result = run(path, args.model, landscape=not args.portrait)
        except Exception as e:  # keep processing remaining images
            print(f"[{path}] error: {e}", file=sys.stderr)
            exit_code = 1
            continue

        stem = Path(path).stem
        print(f"[{path}] {len(result.elements)} elements, "
              f"{result.lines.shape[0]} raw line segments", file=sys.stderr)

        if args.debug_dir:
            _dump_debug(args.debug_dir, stem, result)

        outputs.append(f"% ---- {path} ----\n{result.circuitikz}\n")

    text = "\n".join(outputs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
