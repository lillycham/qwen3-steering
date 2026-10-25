#!/usr/bin/env bash
# Cut a parquet of Wikipedia articles (title, topic, text) into passages for proprio.train.
#   scripts/make-wiki-chunks.sh ARTICLES.parquet [OUT.json] [ARTICLES_TO_SAMPLE]
# duckdb exports a sample of the articles as JSON lines, then proprio.corpus cuts and splits them. With Nix, duckdb
# comes from nixpkgs; without it, have `duckdb` on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
src=${1:?usage: scripts/make-wiki-chunks.sh ARTICLES.parquet [OUT.json] [ARTICLES_TO_SAMPLE]}
out=${2:-runs/proprio/wiki.json}
n=${3:-3000}
tmp=$(mktemp -t wiki-articles.XXXXXX)
trap 'rm -f "$tmp"' EXIT
sql="COPY (SELECT title, topic, substr(text, 1, 40000) AS text FROM read_parquet('$src') USING SAMPLE reservoir($n ROWS) REPEATABLE (0)) TO '$tmp' (FORMAT JSON)"
if command -v duckdb >/dev/null; then duckdb -c "$sql"; else nix shell nixpkgs#duckdb -c duckdb -c "$sql"; fi
mkdir -p "$(dirname "$out")"
if command -v nix >/dev/null; then nix develop -c python -m proprio.corpus build "$tmp" "$out"; else python -m proprio.corpus build "$tmp" "$out"; fi
