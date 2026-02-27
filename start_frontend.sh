#!/usr/bin/env bash
cd "$(dirname "$0")/frontend"
exec ./node_modules/.bin/vite
