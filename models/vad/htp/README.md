# VAD — HTP: N/A (CPU-only by design)

Silero VAD is 1.7 MB and already **sub-millisecond on CPU (~410× real-time)**, and must run
**always-on** to gate the whole pipeline. There is no latency or power benefit to the NPU, so
**VAD stays on CPU**. (This answers "should VAD run on CPU?" — yes.)
