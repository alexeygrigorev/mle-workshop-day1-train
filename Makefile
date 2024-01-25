
.PHONY: tests

tests:
	pipenv run python -m unittest discover -s tests

run: tests
	pipenv run python -m duration_prediction.main \
		--train-month=2023-01 \
		--validation-month=2023-02 \
		--model-output-path=./models/model-2023-01.bin