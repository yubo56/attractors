PYS = $(wildcard [0-9]*.py)
PNGS = $(PYS:.py=.png)

all: $(PNGS)

%.png: %.py
	rm -f $@ && python $<

clean:
	rm -f *.png
