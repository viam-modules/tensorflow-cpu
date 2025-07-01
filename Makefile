setup:
	./build.sh
	
test:
	PYTHONPATH=./src pytest
	
dist/archive.tar.gz:
	tar -czvf dist/archive.tar.gz dist/main

lint:
	pylint --disable=C0114,C0115,C0116,C0103,C0303,C0200,W0719,W4902,W0212,C0301,E1101,E0401,W1201,W0201 src/

