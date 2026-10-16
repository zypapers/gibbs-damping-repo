#!/usr/bin/env bash
# LoG requires the camera-ready PDF to contain Type-1 fonts only. Matplotlib emits
# Type-3 for mathtext regardless of rcParams, and the usetex route needs type1cm.sty
# (texlive-latex-extra), which is not installed. Converting figure text to vector
# outlines leaves no fonts in the figures at all, which satisfies the requirement and
# keeps everything vector for print. Run after src/paper_figures.py.
set -e
cd "$(dirname "$0")/.."
for f in paper/figures/*.pdf; do
  gs -q -dNOPAUSE -dBATCH -dNoOutputFonts -sDEVICE=pdfwrite \
     -dPDFSETTINGS=/prepress -sOutputFile="$f.tmp" "$f"
  mv "$f.tmp" "$f"
  n=$(pdffonts "$f" | tail -n +3 | grep -c . || true)
  echo "  $(basename "$f"): $n embedded fonts"
done
