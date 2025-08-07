SHELL := /bin/bash

.setup: setup.sh
	./setup.sh

.PHONY: setup
setup: .setup

test:
	PYTHONPATH=./src pytest

ifeq ($(OS),Windows_NT)
ACTIVATE = .venv/Scripts/activate
else
ACTIVATE = .venv/bin/activate
endif
dist/main: .setup src/*
	source $(ACTIVATE) && python -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data="./src:src" src/main.py

dist/archive.tar.gz: dist/main
ifeq ($(OS),Windows_NT)
	tar -czvf dist/archive.tar.gz dist/main.exe meta.json
else
	tar -czvf dist/archive.tar.gz dist/main meta.json
endif

lint:
	pylint --disable=C0114,C0115,C0116,C0103,C0303,C0200,W0719,W4902,W0212,C0301,E1101,E0401,W1201,W0201,W0613 src/

