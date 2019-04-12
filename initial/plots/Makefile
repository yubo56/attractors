PYS = $(wildcard [0-9]*.py)
PNGS = $(PYS:.py=.png)

all: $(PNGS)

rm3pkl:
	rm -f *.pkl && touch 3stats.py

%.png: %.py utils.py
	rm -f *$@ && python $<

clean:
	rm -f *.png
