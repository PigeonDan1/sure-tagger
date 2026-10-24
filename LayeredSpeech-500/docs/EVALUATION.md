# Optional Evaluation

The main generation path is offline and does not require sure-tagger, Qwen Omni,
DashScope, OpenAI, or other hosted services.

Evaluation integrations may be added by the user after generation. They must:

1. receive the generated manifest through an explicit input path;
2. keep credentials in environment variables or a secret manager;
3. write outputs under `runs/evaluation/`;
4. record model name, revision, service endpoint, request configuration, and
   result hashes;
5. never write credentials or private absolute paths into the release tree.

The checked-in Ground Truth is model-free and can be regenerated from the final
session manifests and stems without any hosted evaluator.
