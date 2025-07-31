SHELL := /bin/bash

.setup: setup.sh
	./setup.sh

setup: .setup
	
test:
	PYTHONPATH=./src pytest

dist/main: .setup src/*
ifeq ($(OS),Windows_NT)
	source .venv/Scripts/activate && python -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data="./src:src" src/main.py
else
	source .venv/bin/activate && python -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data="./src:src" src/main.py
endif

dist/archive.tar.gz: dist/main
ifeq ($(OS),Windows_NT)
	tar -czvf dist/archive.tar.gz dist/main.exe
else
	tar -czvf dist/archive.tar.gz dist/main
endif

lint:
	pylint --disable=C0114,C0115,C0116,C0103,C0303,C0200,W0719,W4902,W0212,C0301,E1101,E0401,W1201,W0201,W0613 src/

