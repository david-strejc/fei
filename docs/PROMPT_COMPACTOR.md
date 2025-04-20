# Prompt Compactor

The Prompt Compactor is a system for efficiently managing conversation context in Fei. It keeps your coding agents laser-focused and the context window lean, without losing any of the accumulated know-how.

## Why You Need It

* LLM quality **drops** when you blindly add more and more tokens; cost and latency climb too.
* A running code-agent typically needs:
  * the **most recent N turns** verbatim (for local coherence); and
  * an **executive summary** of everything older (for global coherence).

Prompt Compactor automates that trade-off.

## How It Works

The Prompt Compactor consists of four main components:

1. **Recorder**: Appends every *user/assistant* turn to an in-memory log.
2. **Trigger**: Decides *when* to compact based on token count.
3. **Summarizer**: Uses the LLM to both summarize previous conversation and respond to the current message in a single call.
4. **Merger**: Prepends the running **conversation-summary** to the retained verbatim tail and feeds that to the *primary* agent.

### Inline Response Generation

A key feature of Fei's Prompt Compactor is its ability to generate responses inline during compaction. When the conversation reaches the compaction threshold:

1. The system identifies the oldest messages to be summarized
2. It sends these messages along with the current user query to the LLM
3. The LLM is instructed to both summarize the older messages AND respond to the current query
4. The summary is stored for future context, and the response is returned immediately

This approach is more efficient than making separate calls for summarization and response generation.

## Configuration

The Prompt Compactor can be configured through the following settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `compactor.enabled` | Whether to enable compaction | `false` |
| `compactor.tail_k` | Number of recent messages to keep verbatim | `6` |
| `compactor.target_tokens` | Target token count for context | `6000` |
| `compactor.summary_style` | Style for summaries (bullet, narrative, diff, docstring) | `bullet` |
| `compactor.summarize_provider` | Provider to use for summarization | `openai` |
| `compactor.summarize_model` | Model to use for summarization | `gpt-3.5-turbo` |

## Usage

### Command Line

```bash
# Enable prompt compaction
fei --enable-compaction

# Customize compaction settings
fei --enable-compaction --tail-k 8 --target-tokens 8000
```

### Environment Variables

```bash
# Enable prompt compaction
export FEI_COMPACTOR_ENABLED=true

# Customize compaction settings
export FEI_COMPACTOR_TAIL_K=8
export FEI_COMPACTOR_TARGET_TOKENS=8000
export FEI_COMPACTOR_SUMMARY_STYLE=bullet
export FEI_COMPACTOR_SUMMARIZE_PROVIDER=openai
export FEI_COMPACTOR_SUMMARIZE_MODEL=gpt-3.5-turbo
```

### Configuration File

You can also configure the Prompt Compactor in your `~/.fei.ini` file:

```ini
[compactor]
enabled = true
tail_k = 6
target_tokens = 6000
summary_style = bullet
summarize_provider = openai
summarize_model = gpt-3.5-turbo
```

### Example

```bash
# Run the Prompt Compactor example
python examples/prompt_compactor_example.py --enable-compaction
```

## Benefits

* **Reduced Token Usage**: Keep context windows lean by summarizing older messages.
* **Improved Response Quality**: Focus the LLM on the most relevant context.
* **Lower Costs**: Reduce the number of tokens sent to the LLM.
* **Faster Responses**: Fewer tokens means faster processing.
* **Longer Conversations**: Have extended conversations without hitting context limits.

## Advanced Features

* **Hierarchical Summaries**: For long multi-day code refactors, the system can maintain a tree of summaries.
* **Domain-Specific Reducers**: For coding agents, summaries can include diff-style summaries of code changes.
* **Dual-Model Budget Mode**: Use a cheaper model for summarization to reduce costs.

## Implementation Details

The Prompt Compactor is implemented in the `fei/core/prompt_compactor.py` module and integrated with the `ConversationManager` class. It uses the `TokenCounter` utility in `fei/utils/token_counter.py` to count tokens in messages.

When compaction is triggered, the oldest messages are summarized and replaced with a system message containing the summary. The most recent messages (as configured by `tail_k`) are kept verbatim to maintain local coherence.

## Bullet-Proofing

The Prompt Compactor includes several features to ensure reliability:

* **Never Lose Data**: All raw messages are stored in memory.
* **Timeout & Retries**: If summarization fails or times out, the system falls back to a simple concatenation.
* **Auditability**: The system maintains a record of when compaction occurred.
* **Graceful Degradation**: If compaction fails, the system continues with the full conversation.
