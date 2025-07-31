#!/usr/bin/env bash
# setup.sh -- environment bootstrapper for python virtualenv

set -euo pipefail

SUDO=sudo
if ! command -v $SUDO; then
    echo "no sudo on this system, proceeding as current user"
    SUDO=""
fi


if command -v apt-get; then
    $SUDO apt-get -y install python3-venv
    if dpkg -l python3-venv; then
        echo "python3-venv is installed, skipping setup"
    else
        if ! apt info python3-venv; then
            echo "python3-venv package info not found, trying apt update"
            $SUDO apt-get -qq update
        fi
        $SUDO apt-get install -qqy python3-venv
    fi
else
    echo "Skipping tool installation because your platform is missing apt-get."
    echo "If you see failures below, install the equivalent of python3-venv for your system."
fi

source .env
echo "creating virtualenv at $VIRTUAL_ENV"
python3 -m venv $VIRTUAL_ENV

echo "installing dependencies from requirements.txt"
source $VIRTUAL_ENV/bin/activate
pip install --prefer-binary -r requirements.txt -U

touch .setup
