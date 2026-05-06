package io.github.javpower.jpyml.llm.data;

import java.util.Map;

/**
 * A single chat message with role and content.
 */
public record ChatMessage(String role, String content) {

    public static ChatMessage system(String content) { return new ChatMessage("system", content); }
    public static ChatMessage user(String content) { return new ChatMessage("user", content); }
    public static ChatMessage assistant(String content) { return new ChatMessage("assistant", content); }

    public Map<String, String> toMap() {
        return Map.of("role", role, "content", content);
    }
}
