# Native adapters

`OriginalWorker.swift` calls the original app's existing Swift library through
`EG2Transcriber.swiftinterface`. `prepare-original.py` creates the local dynamic
library; `build.sh` compiles both workers. They accept one JSON object per line
on stdin with a local audio `path`, and emit readiness and result objects on
stdout. The Python backend owns their lifetime and serializes calls.

`OpenWorker.swift` uses the preserved upstream audio and pipeline sources under
`open/`. Deployment, model provenance, and endpoint details are in
[`../README.md`](../README.md).
