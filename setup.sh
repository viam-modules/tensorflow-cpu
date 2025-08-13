#!/usr/bin/env bash
# setup.sh -- environment bootstrapper for python virtualenv

set -exuo pipefail

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
# For cloud build, we need to downgrade to tensorflow 2.14.0 which requires python 3.11 or lower
# Since Kongsberg training scripts are written for python 3.10, we will use python 3.10
python3.10 -m venv $VIRTUAL_ENV

echo "installing dependencies from requirements.txt"
if [ -f $VIRTUAL_ENV/bin/activate ]; then # Linux
    source $VIRTUAL_ENV/bin/activate
else
    source $VIRTUAL_ENV/Scripts/activate # Windows
fi
pip install --prefer-binary -r requirements.txt -U

touch .setup
