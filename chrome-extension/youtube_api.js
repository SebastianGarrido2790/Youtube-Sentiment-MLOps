// ==========================================================
// YouTube Data API v3 Helper for Sentiment Chrome Extension
// ==========================================================
//
// Handles paginated retrieval of YouTube comments for a given video ID.
// Enhanced to include timestamps for advanced insights (e.g., trend analysis).
// Works client-side within Chrome extensions (requires YouTube Data API key).
//
// Example usage:
//   import { fetchYouTubeComments } from "./youtube_api.js";
//   const comments = await fetchYouTubeComments("VIDEO_ID", API_KEY, 200);
//
// ==========================================================

/**
 * Fetch comments for a given YouTube video using the official API.
 * Automatically handles pagination and limits. Returns comments with timestamps
 * for compatibility with advanced endpoints (e.g., /predict_with_timestamps).
 *
 * @param {string} videoId - YouTube video ID
 * @param {string} apiKey - YouTube Data API key
 * @param {number} maxResults - Maximum comments to retrieve (default = 100)
 * @returns {Promise<Array<{text: string, timestamp: string}>>} Array of comment objects with text and ISO timestamp
 */
export async function fetchYouTubeComments(videoId, apiKey, maxResults = 100) {
  if (!videoId) throw new Error("Missing YouTube video ID.");
  if (!apiKey) throw new Error("Missing YouTube Data API key.");

  const baseUrl = "https://www.googleapis.com/youtube/v3/commentThreads";
  const comments = [];
  let nextPageToken = null;
  let totalFetched = 0;
  const pageSize = 100; // API max per request

  try {
    while (totalFetched < maxResults) {
      const remaining = Math.min(pageSize, maxResults - totalFetched);
      const url = new URL(baseUrl);
      url.search = new URLSearchParams({
        part: "snippet",
        videoId: videoId,
        key: apiKey,
        maxResults: remaining.toString(),
        pageToken: nextPageToken || "",
        textFormat: "plainText",
        order: "relevance",
      });

      const response = await fetch(url, { method: "GET" });
      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`YouTube API error (${response.status}): ${errText}`);
      }

      const data = await response.json();
      if (!data.items || !Array.isArray(data.items)) break;

      for (const item of data.items) {
        try {
          const text = item.snippet?.topLevelComment?.snippet?.textDisplay?.trim();
          const timestamp = item.snippet?.topLevelComment?.snippet?.publishedAt;
          if (text && timestamp) {
            comments.push({ text, timestamp });
          }
        } catch {
          // Skip malformed items safely
        }
      }

      totalFetched = comments.length;
      nextPageToken = data.nextPageToken || null;

      // Stop if no more pages
      if (!nextPageToken) break;

      // Gentle delay to avoid quota/rate limit issues (especially at high volume)
      await sleep(300);
    }

    console.info(
      `[YouTube API] Retrieved ${comments.length}/${maxResults} comments for video ${videoId}`
    );
    return comments.slice(0, maxResults);
  } catch (err) {
    console.error("[YouTube API] Failed to fetch comments:", err);
    throw new Error("Failed to fetch YouTube comments. " + err.message);
  }
}

/**
 * Sleep helper for throttling between API requests.
 * @param {number} ms
 * @returns {Promise<void>}
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}