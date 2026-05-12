.PHONY: run run-mcq run-rcm download-results clean

run:
	modal run --detach modal_eval_vllm.py --task all

run-mcq:
	modal run --detach modal_eval_vllm.py --task cti-mcq

run-rcm:
	modal run --detach modal_eval_vllm.py --task cti-rcm

download-results:
	mkdir -p results
	modal volume get cybersecqwen-eval-results /eval_results.json --destination results/eval_results.json 2>/dev/null || \
		{ echo "No results volume found. Run 'make run' first."; exit 1; }
	@echo "Results downloaded to results/"

clean:
	rm -rf results/
	modal volume rm cybersecqwen-eval-results --force 2>/dev/null || true
