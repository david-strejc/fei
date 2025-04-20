# Fei Model Optimizer

This document describes the dynamic model optimization mechanism implemented within Fei.

## Purpose

The Model Optimizer aims to automatically select the most suitable Large Language Model (LLM) for a given task based on predefined criteria, primarily focusing on minimizing operational costs while maintaining performance. By leveraging `litellm`, Fei can potentially switch between different providers (like OpenAI, Anthropic, Groq, Google) and models dynamically, choosing the one that offers the best value at the time of the request.

## How it Works

1.  **Integration Point:** The core logic resides in the `fei.core.model_optimizer.ModelOptimizer` class. An instance of this class is created within the `fei.core.assistant.Assistant`.
2.  **Configuration Check:** Before any LLM call (`litellm.completion`), the `Assistant` checks if the optimizer is enabled via the `optimizer.enabled` configuration setting. If disabled, the default configured model (`user.default_model` or provider-specific default) is used.
3.  **Optimal Model Selection:** If enabled, the `Assistant` calls `model_optimizer.get_optimal_model()`. This method performs the following steps:
    *   **Strategy:** Reads the desired `optimizer.strategy` (currently defaults to and only supports `cost`).
    *   **Cost Data:** Fetches model cost information using `litellm.model_cost`. This dictionary provides input and output costs per token for various models known to `litellm`.
    *   **Filtering:**
        *   Removes models listed in `optimizer.model_blacklist`.
        *   If `optimizer.allowed_providers` is set, it removes models from providers not in the list.
    *   **API Key Check:** For each remaining model, it verifies if a corresponding API key is available (checking environment variables like `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc., the generic `LLM_API_KEY`, or specific config entries like `anthropic.api_key`). Models without a configured API key are excluded.
    *   **Cost Comparison:** It calculates a simplified cost (currently based on input cost per 1k tokens) for the filtered models.
    *   **Selection:** It selects the model with the lowest positive calculated cost.
    *   **Fallback:** If no suitable model is found after filtering and cost comparison (e.g., no models with keys, cost data unavailable), it logs a warning and falls back to using the `Assistant`'s originally configured `self.model`.
4.  **Caching:** To avoid repeatedly fetching and calculating model costs, the result of `_find_best_model` is cached using `functools.lru_cache`. The cache is invalidated based on a time interval defined by `optimizer.update_interval` (defaulting to 3600 seconds / 1 hour). This means the "best" model is recalculated periodically.
5.  **LLM Call:** The `Assistant` uses the selected model (either the optimized one or the fallback) for the `litellm.completion` call.

## Configuration

The Model Optimizer's behavior is controlled via the `[optimizer]` section in `~/.fei.ini` or corresponding environment variables.

*   **`enabled` / `FEI_OPTIMIZER_ENABLED`**:
    *   Type: Boolean
    *   Default: `false`
    *   Description: Set to `true` to enable the optimizer.
*   **`strategy` / `FEI_OPTIMIZER_STRATEGY`**:
    *   Type: String
    *   Default: `cost`
    *   Choices: `cost`, `latency`, `cost_performance_ratio`
    *   Description: Defines the criteria for selecting the best model. **Note:** Currently, only the `cost` strategy is implemented. Others will fall back to `cost`.
*   **`allowed_providers` / `FEI_OPTIMIZER_ALLOWED_PROVIDERS`**:
    *   Type: List (Comma-separated string for environment variable)
    *   Default: `[]` (empty list)
    *   Description: If specified, only models from these providers (e.g., `openai`, `groq`) will be considered. An empty list means all providers with available API keys are considered.
*   **`model_blacklist` / `FEI_OPTIMIZER_MODEL_BLACKLIST`**:
    *   Type: List (Comma-separated string for environment variable)
    *   Default: `[]` (empty list)
    *   Description: A list of specific model IDs (e.g., `openai/gpt-3.5-turbo`, `groq/llama3-8b-8192`) to exclude from optimization, regardless of cost or availability.
*   **`update_interval` / `FEI_OPTIMIZER_UPDATE_INTERVAL`**:
    *   Type: Integer
    *   Default: `3600`
    *   Minimum: `60`
    *   Description: The interval in seconds after which the cached optimal model selection is considered stale and recalculated.

**Example `.env` configuration:**

```dotenv
# Enable the optimizer and use the cost strategy
FEI_OPTIMIZER_ENABLED=true
FEI_OPTIMIZER_STRATEGY=cost

# Only consider models from OpenAI and Groq
FEI_OPTIMIZER_ALLOWED_PROVIDERS=openai,groq

# Exclude specific models
FEI_OPTIMIZER_MODEL_BLACKLIST=openai/gpt-3.5-turbo,groq/llama3-8b-8192

# Recalculate the best model every 2 hours
FEI_OPTIMIZER_UPDATE_INTERVAL=7200
```

## Current Limitations

*   **Strategy Implementation:** Only the `cost` strategy is currently functional. `latency` and `cost_performance_ratio` are placeholders.
*   **Cost Calculation:** The cost comparison is simplified, currently only considering the input cost per 1k tokens. It doesn't factor in output costs or potential differences in context window sizes affecting overall cost for specific tasks.
*   **Performance Data:** The optimizer does not currently consider model performance metrics (speed, quality, accuracy).
*   **Availability:** It relies on `litellm.model_cost` for model data and doesn't actively check real-time model/provider availability or uptime.
*   **API Key Granularity:** It assumes a provider-level API key works for all models within that provider (e.g., one `OPENAI_API_KEY` for all OpenAI models).

## Future Enhancements

*   Implement `latency` and `cost_performance_ratio` strategies.
*   Develop a more sophisticated cost model considering input/output costs and context length.
*   Integrate performance benchmarking data.
*   Add checks for real-time model availability.
*   Allow task-specific optimization hints (using the `task_description` parameter).
*   Potentially integrate with external services for more up-to-date model information.
