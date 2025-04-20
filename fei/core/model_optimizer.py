#!/usr/bin/env python3
"""
Model Optimizer for Fei

This module provides mechanisms for dynamically selecting the optimal LLM model
based on criteria like cost, performance, or availability.
"""

import time
import os
import asyncio
from typing import Optional, Dict, List, Tuple, Any
from functools import lru_cache
import litellm
from litellm import completion as litellm_completion # Import completion directly
from litellm import cost_per_token # For more accurate cost calculation

from fei.utils.config import Config
from fei.utils.logging import get_logger

logger = get_logger(__name__)

class ModelOptimizer:
    """
    Selects the best available LLM model based on defined criteria.
    """

    def __init__(self, config: Config):
        """
        Initialize the ModelOptimizer.

        Args:
            config: The Fei configuration object.
        """
        self.config = config
        self.update_interval = self.config.get_int("optimizer.update_interval", 3600)
        self._last_update_time = 0
        self._cached_optimal_model: Optional[Tuple[str, Dict[str, Any]]] = None # Store model and its metrics
        self.evaluation_prompt = self.config.get_string(
            "optimizer.evaluation_prompt",
            "Briefly explain the concept of asynchronous programming in Python." # Default eval prompt
        )

    # --- Caching ---
    # We need an async-compatible cache or manual time-based check
    async def _get_cached_optimal_model_async(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Gets the cached optimal model if it's still valid."""
        current_time = time.time()
        if self._cached_optimal_model and (current_time - self._last_update_time < self.update_interval):
            logger.debug(f"Using cached optimal model: {self._cached_optimal_model[0]}")
            return self._cached_optimal_model
        logger.debug("Optimizer cache expired or empty.")
        return None

    async def _update_cache_async(self, model_id: str, metrics: Dict[str, Any]):
        """Updates the cache with the new best model and timestamp."""
        self._cached_optimal_model = (model_id, metrics)
        self._last_update_time = time.time()
        logger.debug(f"Updated optimizer cache. Best model: {model_id}, Metrics: {metrics}")

    # --- Helper Methods ---
    def _is_api_key_available(self, model_id: str) -> bool:
        """Checks if an API key is configured for the model's provider. (Remains sync)"""
        # Extract provider from model_id (e.g., "openai/gpt-4o" -> "openai")
        # Handle potential prefixes like "groq/"
        provider = model_id.split('/')[0]
        if provider in ["anthropic", "openai", "groq", "google", "brave"]: # Known providers
             # Check using the same logic as ProviderManager._setup_api_key
             env_key = f"{provider.upper()}_API_KEY"
             config_key = f"{provider}.api_key"
             key = (
                 self.config.get(config_key) or
                 os.environ.get(env_key) or
                 os.environ.get("LLM_API_KEY") # Generic fallback
             )
             return bool(key)
        else:
            # Assume unknown providers might work without explicit keys via litellm proxy etc.
            logger.warning(f"Cannot determine API key requirement for unknown provider in model '{model_id}'. Assuming available.")
            return True

    async def _evaluate_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Runs a standard evaluation task on a model and returns metrics."""
        logger.debug(f"Evaluating model: {model_id}")
        metrics = {'latency': float('inf'), 'input_cost': None, 'output_cost': None, 'quality_score': None}
        start_time = time.time()

        try:
            # Use the configured evaluation prompt
            messages = [{"role": "user", "content": self.evaluation_prompt}]

            # Ensure API key is set in environment for litellm (redundant but safe)
            # This relies on _is_api_key_available having been called before evaluation
            provider = model_id.split('/')[0]
            env_key = f"{provider.upper()}_API_KEY"
            if provider in ["anthropic", "openai", "groq", "google"] and not os.environ.get(env_key):
                 config_key = f"{provider}.api_key"
                 key = self.config.get(config_key) or os.environ.get("LLM_API_KEY")
                 if key:
                     os.environ[env_key] = key
                 else:
                     logger.warning(f"API key for {model_id} not found during evaluation setup.")
                     return None # Cannot evaluate without key

            # Run evaluation call
            response = await litellm_completion(
                model=model_id,
                messages=messages,
                max_tokens=150, # Limit output for evaluation
                temperature=0.5, # Consistent temp for eval
                # Note: No tools needed for evaluation prompt
            )
            latency = time.time() - start_time
            metrics['latency'] = latency

            # Get cost using litellm utility
            try:
                 input_cost, output_cost = cost_per_token(model=model_id)
                 metrics['input_cost'] = input_cost
                 metrics['output_cost'] = output_cost
            except Exception as cost_e:
                 logger.warning(f"Could not retrieve cost per token for {model_id}: {cost_e}")
                 # Try falling back to litellm.model_cost if available
                 try:
                     cost_info = litellm.model_cost.get(model_id, {})
                     metrics['input_cost'] = cost_info.get("input_cost_per_token")
                     metrics['output_cost'] = cost_info.get("output_cost_per_token")
                 except Exception:
                     logger.warning(f"Fallback cost retrieval failed for {model_id}")


            # Placeholder for quality scoring
            # TODO: Implement quality scoring logic if needed
            # Example: Check response content, length, or use a judge LLM
            output_text = response.choices[0].message.content if response.choices and response.choices[0].message else ""
            if output_text and len(output_text) > 10: # Very basic quality check
                 metrics['quality_score'] = 1.0
            else:
                 metrics['quality_score'] = 0.0 # Failed basic check

            logger.debug(f"Evaluation result for {model_id}: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}", exc_info=False) # Log less verbosely for eval errors
            metrics['error'] = str(e) # Store error in metrics
            return metrics # Return partial metrics even on error

    async def _find_best_model_async(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Finds the best model based on evaluation and configured strategy."""
        strategy = self.config.get_string("optimizer.strategy", "cost")
        allowed_providers = self.config.get_list("optimizer.allowed_providers", [])
        model_blacklist = self.config.get_list("optimizer.model_blacklist", [])

        candidate_models: List[str] = []
        try:
            # Start with models known to have cost data, as they are likely usable
            candidate_models.extend(litellm.model_cost.keys())
            # TODO: Add models from discovery mechanism here later
        except Exception as e:
            logger.warning(f"Could not get initial model list from litellm.model_cost: {e}")
            # If litellm.model_cost fails, maybe try litellm.model_list? Or rely only on discovery?
            # For now, proceed with an empty list if cost data fails.

        logger.info(f"Starting model evaluation. Strategy: {strategy}. Candidates (initial): {len(candidate_models)}")

        evaluated_models: Dict[str, Dict[str, Any]] = {}
        evaluation_tasks = []

        # Filter and create evaluation tasks
        for model_id in set(candidate_models): # Use set to avoid duplicates
            if model_id in model_blacklist:
                logger.debug(f"Skipping blacklisted model: {model_id}")
                continue

            provider = model_id.split('/')[0]
            if allowed_providers and provider not in allowed_providers:
                logger.debug(f"Skipping model from disallowed provider '{provider}': {model_id}")
                continue

            if not self._is_api_key_available(model_id):
                logger.debug(f"Skipping model due to missing API key: {model_id}")
                continue

            # Add evaluation task
            evaluation_tasks.append(self._evaluate_model(model_id))

        # Run evaluations concurrently
        if not evaluation_tasks:
             logger.warning("No candidate models found for evaluation after filtering.")
             return None

        logger.info(f"Running evaluations for {len(evaluation_tasks)} models...")
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Process results
        valid_results = {}
        for i, result in enumerate(results):
             # Need to get model_id corresponding to the task index - requires careful mapping if filtering changes order
             # Let's re-filter here to be safe, though less efficient
             model_id = None
             eval_index = 0
             for m_id in set(candidate_models):
                 is_blacklisted = m_id in model_blacklist
                 provider = m_id.split('/')[0]
                 is_disallowed = allowed_providers and provider not in allowed_providers
                 has_key = self._is_api_key_available(m_id)
                 if not is_blacklisted and not is_disallowed and has_key:
                     if eval_index == i:
                         model_id = m_id
                         break
                     eval_index += 1

             if model_id is None:
                 logger.error(f"Could not map evaluation result index {i} back to a model ID. Skipping.")
                 continue


             if isinstance(result, Exception):
                 logger.error(f"Evaluation task for {model_id} failed with exception: {result}")
             elif result is None:
                 logger.warning(f"Evaluation for {model_id} returned None.")
             elif result.get('error'):
                 logger.warning(f"Evaluation for {model_id} reported an error: {result['error']}")
             else:
                 # Only consider results without errors and with essential metrics
                 if result.get('latency', float('inf')) != float('inf'): # Ensure latency was measured
                     valid_results[model_id] = result
                 else:
                     logger.warning(f"Evaluation result for {model_id} missing valid latency. Skipping.")


        if not valid_results:
            logger.warning("No models passed evaluation successfully.")
            return None

        # --- Selection based on Strategy ---
        best_model_id: Optional[str] = None
        best_score = float('inf') # Lower is better for cost/latency

        logger.debug(f"Selecting best model from {len(valid_results)} evaluated models using strategy: {strategy}")

        for model_id, metrics in valid_results.items():
            current_score = float('inf')
            input_cost = metrics.get('input_cost') or 0 # Cost per token
            latency = metrics.get('latency', float('inf'))
            # quality = metrics.get('quality_score') # For future use

            if strategy == "cost":
                # Prefer models with known cost, prioritize lower input cost
                if input_cost is not None and input_cost > 0:
                    current_score = input_cost * 1000 # Score based on cost per 1k tokens
                else:
                    logger.debug(f"Model {model_id} missing input cost, cannot use for 'cost' strategy.")
                    continue # Skip if cost is unknown for cost strategy
            elif strategy == "latency":
                current_score = latency
            elif strategy == "cost_performance_ratio":
                 # Example: cost per 1k tokens / quality_score (needs quality score)
                 # Or maybe latency * cost? Lower is better.
                 if input_cost is not None and input_cost > 0 and latency != float('inf'):
                     current_score = (input_cost * 1000) * latency # Simple example: cost * latency
                 else:
                     logger.debug(f"Model {model_id} missing cost or latency, cannot use for 'cost_performance_ratio'.")
                     continue
            else: # Fallback to cost
                if input_cost is not None and input_cost > 0:
                    current_score = input_cost * 1000
                else:
                    continue

            logger.debug(f"Model: {model_id}, Strategy: {strategy}, Score: {current_score}, Metrics: {metrics}")
            if current_score < best_score:
                best_score = current_score
                best_model_id = model_id

        if best_model_id:
            best_metrics = valid_results[best_model_id]
            logger.info(f"Optimizer determined best model ({strategy} strategy): {best_model_id} (Score: {best_score:.4f})")
            return best_model_id, best_metrics
        else:
            logger.warning(f"Optimizer could not select a best model based on strategy '{strategy}' and evaluations.")
            return None

    async def get_optimal_model(
        self,
        current_provider: str, # Keep for potential future context - maybe useful for filtering discovery
        current_model: str, # The default/configured model
        task_description: Optional[str] = None # Optional context for future use
    ) -> str:
        """
        Determines the optimal model to use, considering configuration, evaluation, and cache.

        Args:
            current_provider: The currently configured provider (used for logging/context).
            current_model: The default model configured in the Assistant.
            task_description: Optional description of the task for context-aware optimization.

        Returns:
            The identifier string of the selected optimal model (e.g., "openai/gpt-4o"),
            or the current_model if optimization is disabled or fails.
        """
        if not self.config.get_bool("optimizer.enabled", False):
            logger.debug("Model optimizer is disabled in config.")
            return current_model

        # Check cache first
        cached_result = await self._get_cached_optimal_model_async()
        if cached_result:
            return cached_result[0] # Return model_id from cache

        # If cache missed or expired, recalculate
        logger.info("Recalculating optimal model...")
        try:
            best_model_result = await self._find_best_model_async()

            if best_model_result:
                model_id, metrics = best_model_result
                await self._update_cache_async(model_id, metrics) # Update cache
                return model_id
            else:
                # If optimization failed (e.g., no suitable model found), fall back
                logger.warning(f"Model optimization failed or found no suitable model. Falling back to default: {current_model}")
                # Cache the fallback? Maybe not, allow retry next time.
                return current_model
        except Exception as e:
            logger.error(f"Unexpected error during model optimization recalculation: {e}", exc_info=True)
            # Fallback on any unexpected error
            return current_model
