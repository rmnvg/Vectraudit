# A real promptfoo results file

`results.json` was written by promptfoo 0.125.1 itself, not by hand:

```bash
npx promptfoo@0.125.1 eval -c promptfooconfig.yaml -o results.json --no-cache --no-share --repeat 2
```

The `echo` provider returns the prompt as its output, so the eval runs offline and its
assertions are deterministic. Two prompts × three tests × two repeats = 12 results. The only
change after the run is `config.basePath`, set to `.` to remove the local directory.
