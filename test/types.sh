#!/bin/sh
set -eu

compiler=${1:?Pass the GHC executable}

compile() {
    "$compiler" -XGHC2021 -XNoGeneralizedNewtypeDeriving -Wall -Werror -fno-code -package invar "$1"
}

for fixture in test/accept/*.hs; do
    compile "$fixture"
done

for fixture in test/reject/*.hs; do
    expected=$(sed -n 's/^-- Reject: //p' "$fixture")
    test -n "$expected"
    if diagnostic=$(compile "$fixture" 2>&1); then
        printf 'Unexpectedly accepted: %s\n' "$fixture" >&2
        exit 1
    fi
    printf '%s\n' "$diagnostic"
    if ! printf '%s\n' "$diagnostic" | grep -F -- "$expected" >/dev/null; then
        printf 'Unexpected diagnostic: %s\n' "$fixture" >&2
        exit 1
    fi
done
