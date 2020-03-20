#!bin/bash

for filename in *.xyz; do
	[ -f "$filename" ] || continue
	mv "$filename" "${filename//orca/}"
done
