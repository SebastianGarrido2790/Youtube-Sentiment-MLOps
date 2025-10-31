// ==========================================================
// YouTube Sentiment Chrome Extension - Enhanced Insights
// ==========================================================

import { fetchYouTubeComments } from "./youtube_api.js";

// ========================
// CONFIGURATION
// ========================
const API_KEY = "AIzaSyA3cnCtBOXx_6G8zvxm3Y-OFpjRWD7I_VU";  // Replace with your YouTube API key
const INSIGHTS_URL = "http://127.0.0.1:8001";                   // Insights API base URL (port 8001)

// ========================
// DOM ELEMENTS
// ========================
const analyzeBtn = document.getElementById("analyzeBtn");
const numCommentsInput = document.getElementById("numComments");
const loadingEl = document.getElementById("loading");
const resultsEl = document.getElementById("results");
const totalCountEl = document.getElementById("totalCount");
const breakdownText = document.getElementById("breakdownText");
const pieChartImg = document.getElementById("pieChartImg");     // <img> for pie chart
const legendEl = document.getElementById("legend");
const errorEl = document.getElementById("error");
const backendUrlEl = document.getElementById("backendUrl");
const loadingTextEl = document.getElementById("loadingText");
const videoIdEl = document.getElementById("videoId");           // Video ID display

// Summary metrics
const totalCommentsSummaryEl = document.getElementById("totalCommentsSummary");
const uniqueCommentersEl = document.getElementById("uniqueCommenters");
const avgLenEl = document.getElementById("avgLen");
const avgSentimentEl = document.getElementById("avgSentiment");

// Advanced insights elements (assume added to HTML)
const trendGraphImg = document.getElementById("trendGraphImg"); // <img> for trend graph
const wordcloudImg = document.getElementById("wordcloudImg");   // <img> for wordcloud
const topCommentsEl = document.getElementById("topComments");   // <ul> or <div> for top 25

backendUrlEl.textContent = INSIGHTS_URL;

// ========================
// UI HELPERS
// ========================
function showLoading(text = "Analyzing comments...") {
  loadingTextEl.innerText = text;
  loadingEl.classList.remove("hidden");
  resultsEl.classList.add("hidden");
  errorEl.classList.add("hidden");
  analyzeBtn.disabled = true;
}

function hideLoading() {
  loadingEl.classList.add("hidden");
  analyzeBtn.disabled = false;
}

function showError(msg) {
  errorEl.innerText = msg;
  errorEl.classList.remove("hidden");
  resultsEl.classList.add("hidden");
  loadingEl.classList.add("hidden");
  analyzeBtn.disabled = false;
}

function displayImage(imgEl, blob) {
  const url = URL.createObjectURL(blob);
  imgEl.src = url;
  imgEl.style.display = "block";
}

// ========================
// YOUTUBE COMMENT FETCH
// ========================
function getVideoIdFromUrl(url) {
  try {
    const params = new URL(url).searchParams;
    return params.get("v");
  } catch {
    return null;
  }
}

async function getCommentsFromAPI(maxResults = 100) {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (!tabs || !tabs.length) return reject("No active YouTube tab found.");
      const tab = tabs[0];
      const videoId = getVideoIdFromUrl(tab.url);
      if (!videoId) return reject("No video detected. Please open a YouTube video.");

      // Display Video ID
      if (videoIdEl) videoIdEl.textContent = videoId;

      try {
        const comments = await fetchYouTubeComments(videoId, API_KEY, maxResults);
        resolve(comments);  // Now [{text, timestamp}]
      } catch (err) {
        reject(err);
      }
    });
  });
}

// ========================
// INSIGHTS API CALLS
// ========================
async function callPredictWithTimestamps(commentsData) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000); // 30s timeout

  try {
    const resp = await fetch(`${INSIGHTS_URL}/predict_with_timestamps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: commentsData }),
      signal: controller.signal,
    });

    clearTimeout(timeout);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Insights API error (${resp.status}): ${text}`);
    }

    return await resp.json();
  } catch (err) {
    if (err.name === "AbortError") throw new Error("API timed out (30s).");
    throw err;
  }
}

async function generateChart(sentimentCounts) {
  const resp = await fetch(`${INSIGHTS_URL}/generate_chart`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentiment_counts: sentimentCounts }),
  });
  if (!resp.ok) throw new Error("Failed to generate pie chart.");
  return await resp.blob();
}

async function generateWordcloud(comments) {
  const resp = await fetch(`${INSIGHTS_URL}/generate_wordcloud`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ comments }),
  });
  if (!resp.ok) throw new Error("Failed to generate wordcloud.");
  return await resp.blob();
}

async function generateTrendGraph(sentimentData) {
  const resp = await fetch(`${INSIGHTS_URL}/generate_trend_graph`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentiment_data: sentimentData }),
  });
  if (!resp.ok) throw new Error("Failed to generate trend graph.");
  return await resp.blob();
}

// ========================
// ANALYTICS HELPERS
// ========================
function normalizeSentiments(sentiments) {
  // Map numeric (-1,0,1) to strings
  return sentiments.map(s => {
    if (s === -1) return "Negative";
    if (s === 0) return "Neutral";
    if (s === 1) return "Positive";
    return "Unknown";
  });
}

function computeSentimentCounts(sentiments) {
  const counts = { Positive: 0, Neutral: 0, Negative: 0 };
  sentiments.forEach(s => {
    if (s === 1) counts.Positive++;
    else if (s === 0) counts.Neutral++;
    else if (s === -1) counts.Negative++;
  });
  return counts;
}

function updateLegendAndBreakdown(counts) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const labels = ["Positive", "Neutral", "Negative"];
  legendEl.innerHTML = "";

  labels.forEach(label => {
    const count = counts[label] || 0;
    const pct = total ? ((count / total) * 100).toFixed(1) : "0.0";
    const color = label === "Positive" ? "#10b981" : label === "Neutral" ? "#9ca3af" : "#ef4444";

    const item = document.createElement("div");
    item.classList.add("item");
    item.innerHTML = `
      <span class="swatch" style="background:${color}"></span>
      ${label}: <strong>${count}</strong> (${pct}%)
    `;
    legendEl.appendChild(item);
  });

  breakdownText.innerHTML = `
    Positive <strong>${counts["Positive"] || 0}</strong> —
    Neutral <strong>${counts["Neutral"] || 0}</strong> —
    Negative <strong>${counts["Negative"] || 0}</strong>
  `;
}

function displayTopComments(results, limit = 25) {
  if (!topCommentsEl) return;
  topCommentsEl.innerHTML = "";
  const sorted = results.slice(0, limit).map(r => ({
    ...r,
    label: normalizeSentiments([r.sentiment])[0]
  }));
  sorted.forEach((item, i) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <strong>${i + 1}.</strong> ${item.comment.substring(0, 100)}... 
      <span style="color: ${item.sentiment === 1 ? '#10b981' : item.sentiment === 0 ? '#9ca3af' : '#ef4444'}">
        (Sentiment: ${item.sentiment})
      </span>
    `;
    topCommentsEl.appendChild(li);
  });
}

// ========================
// MAIN WORKFLOW
// ========================
analyzeBtn.addEventListener("click", async () => {
  showLoading("Fetching comments from YouTube...");

  try {
    const limit = Math.max(1, Number(numCommentsInput.value) || 100);

    // Step 1: Fetch comments with timestamps
    const commentsData = await getCommentsFromAPI(limit);
    if (!commentsData || commentsData.length === 0)
      throw new Error("No comments retrieved.");

    showLoading(`Analyzing ${commentsData.length} comments...`);

    // Step 2: Predict with timestamps
    const resp = await callPredictWithTimestamps(commentsData);
    const results = resp.results || [];
    if (!results.length) throw new Error("No predictions returned.");

    // Step 3: Extract sentiments (numeric -1/0/1)
    const sentiments = results.map(r => r.sentiment);
    const stringSentiments = normalizeSentiments(sentiments);

    // Step 4: Compute metrics
    const counts = computeSentimentCounts(sentiments);
    const totalComments = commentsData.length;
    const uniqueCommenters = new Set(commentsData.map(c => c.author || "")).size; // Assume author if available; adjust if needed
    const totalWords = commentsData.reduce(
      (sum, c) => sum + c.text.split(/\s+/).filter(w => w.length > 0).length,
      0
    );
    const avgCommentLength = (totalWords / totalComments).toFixed(2);

    const sentimentMap = { Positive: 1, Neutral: 0, Negative: -1 };
    const avgSentimentRaw = stringSentiments
      .map(p => sentimentMap[p] ?? 0)
      .reduce((a, b) => a + b, 0) / totalComments;
    const avgSentimentScore = (((avgSentimentRaw + 1) / 2) * 10).toFixed(2);

    // Step 5: Update basic UI
    totalCountEl.innerText = totalComments;
    totalCommentsSummaryEl.innerText = totalComments;
    uniqueCommentersEl.innerText = uniqueCommenters;
    avgLenEl.innerText = avgCommentLength;
    avgSentimentEl.innerText = avgSentimentScore;

    updateLegendAndBreakdown(counts);

    // Step 6: Generate and display visuals
    showLoading("Generating visuals...");

    // Pie Chart
    const sentimentCounts = {
      "-1": counts.Negative,
      "0": counts.Neutral,
      "1": counts.Positive
    };
    const chartBlob = await generateChart(sentimentCounts);
    displayImage(pieChartImg, chartBlob);

    // Wordcloud
    const texts = commentsData.map(c => c.text);
    const wordcloudBlob = await generateWordcloud(texts);
    displayImage(wordcloudImg, wordcloudBlob);

    // Trend Graph
    const sentimentData = results.map(r => ({ sentiment: r.sentiment, timestamp: r.timestamp }));
    const trendBlob = await generateTrendGraph(sentimentData);
    displayImage(trendGraphImg, trendBlob);

    // Top Comments
    displayTopComments(results);

    resultsEl.classList.remove("hidden");
    loadingTextEl.innerText = "Analysis complete!";
  } catch (err) {
    console.error("[YouTube Sentiment] Error:", err);
    showError(err.message || String(err));
  } finally {
    hideLoading();
  }
});