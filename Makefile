.PHONY: test seed report clean-report

test:
	PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v

seed:
	PYTHONDONTWRITEBYTECODE=1 python3 -B -m ralphton_icml run-seed

report:
	$(MAKE) -C report report.pdf

clean-report:
	$(MAKE) -C report clean
