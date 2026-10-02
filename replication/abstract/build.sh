#!/bin/sh
# Build the SSAC27 abstract package from this folder: refresh the exhibit copies, rebuild the
# composite figure, compile twice, keep the PDF beside the source and print the word counts.
#
#   sh abstract/SSAC27_submission/build.sh          (from anywhere; needs pdflatex, pdftotext, python3)
#
# What it does not do: update the numbers. The composite figure's rows are typed into
# make_submission_figure.py (ROWS) and the Results sentences into SSAC27_abstract.tex; refresh both
# from the newest docs/FIRST_RESULT_*.md before the final build.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$HERE"
TEX=SSAC27_abstract.tex

# 1. Exhibits: every figures/... path the uncommented source includes is copied from paper/figures
#    when it exists there (the composite figure is produced here in step 2, so it is skipped).
USED=$(grep -v '^[[:space:]]*%' "$TEX" | grep -oE 'figures/[A-Za-z0-9_./-]+' | sort -u)
mkdir -p figures
for f in $USED; do
    src="$ROOT/paper/figures/$(basename "$f")"
    if [ -f "$src" ]; then cp "$src" "$f"; echo "copied  $f  <-  paper/figures"; fi
done
for f in $USED; do [ -f "$f" ] || { echo "MISSING $f (not in figures/ nor in paper/figures)"; exit 1; }; done

# 2. Composite figure (both themes; also writes figures/learning_evidence.csv), and its CRPS variant
#    (fig_learning_and_evidence_crps*, tables in figures/evidence_crps*.csv; CRPS_COMBINE = mean (default) or mixture).
python3 make_submission_figure.py | grep -v '^Text:'
python3 make_evidence_crps_figure.py --combine "${CRPS_COMBINE:-mean}" --show-training | grep -E 'words including'

# 3. Compile twice into build/ (clutter stays there); the PDF is copied beside the source.
mkdir -p build
for i in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$TEX" > build/compile.log 2>&1 \
        || { echo "COMPILE FAILED:"; grep -n -A6 '^!' build/compile.log | head -40; exit 1; }
done
cp build/SSAC27_abstract.pdf SSAC27_abstract.pdf
echo "built   SSAC27_abstract.pdf: $(pdfinfo SSAC27_abstract.pdf | awk '/Pages/{print $2}') pages, $(grep -c '^!' build/compile.log) errors, $(grep -c 'Overfull' build/compile.log) overfull boxes"

# 4. Counts. The rule is fewer than 500 words including the title; figure-internal text is
#    reported separately so the strictest reading is visible too.
if command -v texcount > /dev/null 2>&1; then
    echo "texcount (body + captions): $(texcount -brief -sum "$TEX" 2>/dev/null | tail -1 | cut -d: -f1)"
fi
ALL=$(pdftotext SSAC27_abstract.pdf - | wc -w | tr -d ' ')
FIG=0
for f in $USED; do
    case "$f" in *.pdf) n=$(pdftotext "$f" - 2>/dev/null | wc -w | tr -d ' '); FIG=$((FIG + n));; esac
done
echo "printed words: $ALL in all, $((ALL - FIG)) outside the figures (title, authors, body, captions, references, link), $FIG inside the figures"

# 5. Anything in figures/ that the source no longer includes (informational).
for f in figures/*; do
    case "$(basename "$f")" in fig_learning_and_evidence*|learning_evidence.csv|evidence_source.csv|evidence_crps*.csv) continue;; esac
    echo "$USED" | grep -qx "$f" || echo "unused  $f (not included by the current source)"
done
