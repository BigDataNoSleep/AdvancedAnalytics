const form = document.querySelector("#search-form");
const queryField = document.querySelector("#query");
const submitButton = document.querySelector("#submit-button");
const statusNode = document.querySelector("#status");
const answerNode = document.querySelector("#answer");
const matchesNode = document.querySelector("#matches");
const matchCountNode = document.querySelector("#match-count");
const matchTemplate = document.querySelector("#match-template");
const pipelineLogNode = document.querySelector("#pipeline-log");
const pipelineLogDetails = document.querySelector("#pipeline-log-details");

function setLoadingState(isLoading, message) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Searching..." : "Find matches";
  statusNode.textContent = message;
}

function addLogEntry(msg) {
  // Remove placeholder if present
  const placeholder = pipelineLogNode.querySelector(".log-placeholder");
  if (placeholder) placeholder.remove();

  const p = document.createElement("p");
  p.className = "log-entry";
  p.textContent = msg;
  pipelineLogNode.appendChild(p);
  pipelineLogNode.scrollTop = pipelineLogNode.scrollHeight;
}

function renderMatches(matches) {
  matchesNode.innerHTML = "";
  matchCountNode.textContent = `${matches.length} game${matches.length === 1 ? "" : "s"}`;

  if (!matches.length) {
    matchesNode.innerHTML = "<p class='empty'>No matches returned.</p>";
    return;
  }

  for (const match of matches) {
    const fragment = matchTemplate.content.cloneNode(true);
    const image = fragment.querySelector(".match-image");
    const title = fragment.querySelector(".match-title");
    const score = fragment.querySelector(".match-score");
    const description = fragment.querySelector(".match-description");
    const meta = fragment.querySelector(".match-meta");
    const tags = fragment.querySelector(".match-tags");
    const link = fragment.querySelector(".match-link");

    image.src = match.header_image || "";
    image.alt = match.name;
    image.loading = "lazy";
    title.textContent = match.name;
    score.textContent = `score ${match.score}`;
    description.textContent = match.short_description || "No short description available.";

    const genreText = (match.genres || []).slice(0, 3).join(", ") || "Unknown genre";
    const platformText = Object.entries(match.platforms || {})
      .filter(([, enabled]) => enabled)
      .map(([platform]) => platform)
      .join(", ") || "No platform data";
    meta.textContent = `${genreText} | ${match.release_date || "Unknown release date"} | ${platformText}`;

    for (const tag of (match.tags || []).slice(0, 6)) {
      const pill = document.createElement("span");
      pill.className = "tag";
      pill.textContent = tag;
      tags.appendChild(pill);
    }

    link.href = match.store_page;
    matchesNode.appendChild(fragment);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryField.value.trim();
  if (!query) {
    setLoadingState(false, "Enter a description first.");
    return;
  }

  setLoadingState(true, "Querying local RAG pipeline...");
  answerNode.innerHTML = "<em>Thinking...</em>";
  matchesNode.innerHTML = "";
  matchCountNode.textContent = "0 games";
  pipelineLogNode.innerHTML = "";
  pipelineLogDetails.open = true;
  addLogEntry("⏳ Starting pipeline...");

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Request failed");
    }

    // Read the SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from the buffer
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // Keep incomplete event in buffer

      for (const part of parts) {
        if (!part.trim()) continue;

        const lines = part.split("\n");
        let eventType = "message";
        let data = "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7);
          } else if (line.startsWith("data: ")) {
            data = line.slice(6);
          }
        }

        if (!data) continue;

        try {
          const parsed = JSON.parse(data);

          if (eventType === "log") {
            addLogEntry(parsed.log);
          } else if (eventType === "result") {
            // Render the answer as markdown
            if (typeof marked !== "undefined" && marked.parse) {
              answerNode.innerHTML = marked.parse(parsed.answer || "");
            } else {
              answerNode.textContent = parsed.answer;
            }
            renderMatches(parsed.matches || []);
            setLoadingState(false, `Done. Indexed ${parsed.meta.indexed_games} games.`);
          }
        } catch (e) {
          console.warn("Failed to parse SSE data:", data, e);
        }
      }
    }
  } catch (error) {
    answerNode.textContent = error.message;
    addLogEntry(`❌ Error: ${error.message}`);
    renderMatches([]);
    setLoadingState(false, "Search failed.");
  }
});
