const nameInput = document.getElementById("name-input");
const translateForm = document.getElementById("translate-form");
const queryInput = document.getElementById("query-input");
const translateBtn = document.getElementById("translate-btn");
const resultsEl = document.getElementById("results");
const resultTemplate = document.getElementById("result-template");

const contributeForm = document.getElementById("contribute-form");
const contribMaithili = document.getElementById("contrib-maithili");
const contribEnglish = document.getElementById("contrib-english");
const contributeBtn = document.getElementById("contribute-btn");
const contributeStatus = document.getElementById("contribute-status");

const logList = document.getElementById("log-list");
const logCorrectionTemplate = document.getElementById("log-correction-template");
const logContributionTemplate = document.getElementById("log-contribution-template");

const NAME_KEY = "maithili-ai:name";
nameInput.value = localStorage.getItem(NAME_KEY) || "";
nameInput.addEventListener("input", () => {
  localStorage.setItem(NAME_KEY, nameInput.value.trim());
});

function currentName() {
  return nameInput.value.trim() || "anonymous";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.textContent;
}

function formatWhen(sqliteTimestamp) {
  const date = new Date(sqliteTimestamp.replace(" ", "T") + "Z");
  if (isNaN(date.getTime())) return sqliteTimestamp;
  return date.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  });
}

async function loadLog() {
  try {
    const [correctionsRes, contributionsRes] = await Promise.all([
      fetch("/corrections"),
      fetch("/contributions"),
    ]);
    if (!correctionsRes.ok || !contributionsRes.ok) throw new Error("failed to load log");
    const corrections = (await correctionsRes.json()).map((c) => ({ ...c, _kind: "correction" }));
    const contributions = (await contributionsRes.json()).map((c) => ({ ...c, _kind: "contribution" }));

    const entries = [...corrections, ...contributions].sort(
      (a, b) => new Date(b.created_at) - new Date(a.created_at)
    );

    logList.innerHTML = "";
    if (entries.length === 0) {
      logList.innerHTML = '<p class="log-empty">Nothing taught yet — correct a translation or add a phrase above.</p>';
      return;
    }

    for (const entry of entries) {
      if (entry._kind === "correction") {
        const node = logCorrectionTemplate.content.cloneNode(true);
        node.querySelector(".log-by").textContent = entry.corrected_by;
        node.querySelector(".log-when").textContent = formatWhen(entry.created_at);
        node.querySelector(".log-query").textContent = entry.original_query || "";
        node.querySelector(".log-wrong").textContent = entry.wrong_response;
        node.querySelector(".log-correct").textContent = entry.correct_response;
        const explanationEl = node.querySelector(".log-explanation");
        if (entry.explanation) {
          explanationEl.textContent = entry.explanation;
        } else {
          explanationEl.remove();
        }
        logList.appendChild(node);
      } else {
        const node = logContributionTemplate.content.cloneNode(true);
        node.querySelector(".log-by").textContent = entry.contributor;
        node.querySelector(".log-when").textContent = formatWhen(entry.created_at);
        node.querySelector(".log-phrase").textContent = entry.text_maithili;
        const englishEl = node.querySelector(".log-phrase-english");
        if (entry.text_english) {
          englishEl.textContent = entry.text_english;
        } else {
          englishEl.remove();
        }
        logList.appendChild(node);
      }
    }
  } catch (err) {
    logList.innerHTML = `<p class="log-empty">Couldn't load training log: ${escapeHtml(err.message)}</p>`;
  }
}

loadLog();

translateForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  translateBtn.disabled = true;
  translateBtn.textContent = "translating...";

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: query }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderResult(query, data);
    queryInput.value = "";
  } catch (err) {
    alert(`Translation failed: ${err.message}`);
  } finally {
    translateBtn.disabled = false;
    translateBtn.textContent = "Translate";
  }
});

function renderResult(query, data) {
  const node = resultTemplate.content.cloneNode(true);
  const card = node.querySelector(".result-card");

  card.querySelector(".query").textContent = query;
  card.querySelector(".answer").textContent = data.response;

  const docs = data.retrieved_docs || [];
  const summary = card.querySelector(".trace summary");
  summary.textContent = docs.length
    ? `why the ai said this (${docs.length} source${docs.length === 1 ? "" : "s"} retrieved)`
    : "why the ai said this (no sources retrieved — base model knowledge only)";

  const traceList = card.querySelector(".trace-list");
  docs.forEach((doc) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="doc-type">${escapeHtml(doc.type)}</span> ` +
      `<span class="doc-meta">priority=${doc.priority.toFixed(1)} score=${doc.similarity.toFixed(3)} source="${escapeHtml(doc.source)}"</span>` +
      `<span class="doc-content">${escapeHtml(doc.content)}</span>`;
    traceList.appendChild(li);
  });

  const correctToggle = card.querySelector(".correct-toggle");
  const correctForm = card.querySelector(".correct-form");
  const correctInput = card.querySelector(".correct-input");
  const correctCancel = card.querySelector(".correct-cancel");
  const correctionResult = card.querySelector(".correction-result");

  correctToggle.addEventListener("click", () => {
    correctForm.hidden = !correctForm.hidden;
    if (!correctForm.hidden) correctInput.focus();
  });

  correctCancel.addEventListener("click", () => {
    correctForm.hidden = true;
  });

  correctForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const correctText = correctInput.value.trim();
    if (!correctText) return;

    const submitBtn = correctForm.querySelector(".correct-submit");
    submitBtn.disabled = true;
    submitBtn.textContent = "saving...";

    try {
      const res = await fetch("/correct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: data.conversation_id,
          correct_response: correctText,
          corrected_by: currentName(),
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();

      correctForm.hidden = true;
      correctionResult.hidden = false;
      correctionResult.innerHTML =
        `<span class="label">saved — will outrank the corpus for similar queries</span>${escapeHtml(result.explanation || "")}`;
      loadLog();
    } catch (err) {
      alert(`Saving correction failed: ${err.message}`);
      submitBtn.disabled = false;
      submitBtn.textContent = "save correction";
    }
  });

  resultsEl.prepend(node);
}

contributeForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const maithili = contribMaithili.value.trim();
  const english = contribEnglish.value.trim();
  if (!maithili) return;

  contributeBtn.disabled = true;
  contributeBtn.textContent = "adding...";
  contributeStatus.textContent = "";
  contributeStatus.classList.remove("error");

  try {
    const res = await fetch("/contribute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text_maithili: maithili,
        text_english: english || null,
        contributor: currentName(),
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    contributeStatus.textContent = "added — it'll show up in retrieval trace for related queries.";
    contribMaithili.value = "";
    contribEnglish.value = "";
    loadLog();
  } catch (err) {
    contributeStatus.textContent = `Failed: ${err.message}`;
    contributeStatus.classList.add("error");
  } finally {
    contributeBtn.disabled = false;
    contributeBtn.textContent = "Add";
  }
});
