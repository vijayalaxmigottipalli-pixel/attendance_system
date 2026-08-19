#!/usr/bin/env bash
# exit on first error so a failed step doesn't silently continue
set -o errexit

pip install --upgrade pip

# Install cmake FIRST and separately — dlib's build needs a working cmake
# executable available before it starts compiling, and installing it as
# part of the same requirements.txt pass isn't guaranteed to order correctly.
pip install cmake

# Now install everything else, including dlib (which will compile from
# source here — this step can take several minutes, that's expected).
pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate