.PHONY: all clean

all: output/aff.parquet

clean:
	rm output/*

output/aff.parquet: src/import.py \
	input/Algerian_forest_fires_dataset_UPDATE.csv
	python src/import.py --input=input/Algerian_forest_fires_dataset_UPDATE.csv \
		--output=output/aff.parquet

