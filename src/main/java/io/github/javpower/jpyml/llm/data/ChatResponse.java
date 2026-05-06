package io.github.javpower.jpyml.llm.data;

/**
 * Response from LLM chat inference.
 */
public class ChatResponse {

    private final String content;
    private final int promptTokens;
    private final int completionTokens;
    private final String finishReason;

    public ChatResponse(String content, int promptTokens, int completionTokens, String finishReason) {
        this.content = content;
        this.promptTokens = promptTokens;
        this.completionTokens = completionTokens;
        this.finishReason = finishReason;
    }

    public String getContent() { return content; }
    public int getPromptTokens() { return promptTokens; }
    public int getCompletionTokens() { return completionTokens; }
    public String getFinishReason() { return finishReason; }

    @Override
    public String toString() {
        return "ChatResponse{content='" + content + "', tokens=" + (promptTokens + completionTokens) + "}";
    }
}
