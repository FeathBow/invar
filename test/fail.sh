#!/bin/sh
set -eu

case "$1" in
    --cache=*) directory=${1#--cache=} ;;
    *) exit 2 ;;
esac

IFS= read -r invocation
printf '%s\n' "$invocation" >> "$directory/calls"
exit 7
