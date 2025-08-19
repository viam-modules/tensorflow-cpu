#!/usr/bin/env bash
# setup.sh -- environment bootstrapper for python virtualenv

set -exuo pipefail

SUDO=sudo
if ! command -v $SUDO; then
    echo "no sudo on this system, proceeding as current user"
    SUDO=""
fi


if command -v apt-get; then
    $SUDO apt-get -y install python3.10 python3.10-venv
    if dpkg -l python3.10-venv; then
        echo "python3.10-venv is installed, skipping setup"
    else
        if ! apt info python3.10-venv; then
            echo "python3.10-venv package info not found, trying apt update"
            $SUDO apt-get -qq update
        fi
        $SUDO apt-get install -qqy python3.10-venv
    fi
else
    echo "Skipping tool installation because your platform is missing apt-get."
    echo "If you see failures below, install the equivalent of python3-venv for your system."
fi

source .env
echo "creating virtualenv at $VIRTUAL_ENV"
# WARNING: As of August 2025, Tensorflow is not available in Python 3.13. If you get stuck here,
# consider using python3.12 instead.
python3.10 -m venv $VIRTUAL_ENV

echo "installing dependencies from requirements.txt"
if [ -f $VIRTUAL_ENV/bin/activate ]; then # Linux
    source $VIRTUAL_ENV/bin/activate
else
    source $VIRTUAL_ENV/Scripts/activate # Windows
fi
pip install --prefer-binary -r requirements.txt -U

touch .setup
