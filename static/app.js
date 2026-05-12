/* PersonaAI – Frontend Logic v3 (Dashboard Edition) */

const API = '';
const AUTH_TOKEN = 'testcase1234';

// ── Active Profile (auto-fill) ───────────────────────
let activeProfileId = null;

function setActiveProfile(id) {
  activeProfileId = id;
  // Auto-fill all profile ID fields
  const fields = ['gen-user-id', 'pred-user-id', 'fb-user-id', 'dash-user-id'];
  fields.forEach(fieldId => {
    const el = document.getElementById(fieldId);
    if (el) el.value = id;
  });
}

// ── Tabs ─────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    tab.setAttribute('aria-selected', 'true');
    document.getElementById(tab.dataset.tab).classList.add('active');
  });
});

// ── Helpers ──────────────────────────────────────────
function showToast(message, type = 'success') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = `toast ${type} show`;
  setTimeout(() => { toast.classList.remove('show'); }, 3500);
}

function setLoading(btn, loading) {
  btn.disabled = loading;
  if (loading) {
    btn.dataset.originalText = btn.textContent;
    btn.textContent = 'Processing…';
    btn.classList.add('loading');
  } else {
    btn.textContent = btn.dataset.originalText || btn.textContent;
    btn.classList.remove('loading');
  }
}

async function apiCall(url, body, method = 'POST') {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${AUTH_TOKEN}`
    },
  };
  if (body && method === 'POST') {
    options.body = JSON.stringify(body);
  }
  const res = await fetch(API + url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
  return data;
}

function renderField(label, value) {
  return `<div class="result-field">
    <div class="result-label">${label}</div>
    <div class="result-value">${value}</div>
  </div>`;
}

function renderTags(items) {
  if (!items || !items.length) return '<span class="result-value">—</span>';
  return `<div class="tag-list">${items.map(t => `<span class="tag">${t}</span>`).join('')}</div>`;
}

// ── Copy to clipboard ────────────────────────────────
function copyPost(btn) {
  const text = btn.closest('.result').querySelector('.post-content').textContent;
  navigator.clipboard.writeText(text).then(() => {
    btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    showToast('Copied to clipboard!');
    setTimeout(() => {
      btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
    }, 2000);
  });
}

// ── Dashboard ────────────────────────────────────────
document.getElementById('dashboard-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('.btn');
  const result = document.getElementById('dashboard-result');
  setLoading(btn, true);
  result.classList.add('hidden');

  try {
    const userId = parseInt(document.getElementById('dash-user-id').value);

    // Fetch brand profile
    const brandData = await apiCall(`/dashboard/${userId}`, null, 'GET');

    // Set active profile and auto-fill all tabs
    setActiveProfile(userId);

    let html = `
      <!-- Brand Overview -->
      <div class="dash-brand-section">
        <div class="profile-header">
          <div class="profile-badge">${brandData.name ? brandData.name.charAt(0).toUpperCase() : '?'}</div>
          <div class="profile-info">
            <div class="profile-name">${brandData.name}</div>
            <div class="profile-role">${brandData.role} · ${brandData.industry}</div>
          </div>
          <div class="dash-profile-id">Profile #${userId}</div>
        </div>
        ${renderField('Tone', brandData.tone || '—')}
        ${renderField('Positioning', brandData.positioning_summary || '—')}
        <div class="result-field">
          <div class="result-label">Content Themes</div>
          ${renderTags(brandData.content_themes)}
        </div>
      </div>

      <!-- Stats Overview -->
      <div class="dash-stats-section">
        <h3>Performance Overview</h3>
        <div class="history-summary">
          <div class="stat-card">
            <div class="stat-number">${brandData.total_posts}</div>
            <div class="stat-label">Total Posts</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${brandData.total_likes}</div>
            <div class="stat-label">Total Likes</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${brandData.total_comments}</div>
            <div class="stat-label">Total Comments</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${brandData.total_shares}</div>
            <div class="stat-label">Total Shares</div>
          </div>
        </div>
        <div class="history-summary">
          <div class="stat-card">
            <div class="stat-number">${brandData.avg_likes}</div>
            <div class="stat-label">Avg Likes</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${brandData.avg_comments}</div>
            <div class="stat-label">Avg Comments</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${brandData.avg_shares}</div>
            <div class="stat-label">Avg Shares</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${brandData.best_topic ? '⭐' : '—'}</div>
            <div class="stat-label">Best Topic</div>
          </div>
        </div>
        ${brandData.best_topic ? renderField('Top Performing Topic', brandData.best_topic) : ''}
      </div>

      <!-- Auto-fill Notice -->
      <div class="dash-autofill-notice">
        ✅ Profile ID #${userId} auto-filled across all tabs. You can now switch to any tab without re-entering your ID.
      </div>
    `;

    // Post History section
    if (brandData.posts && brandData.posts.length > 0) {
      html += `
        <div class="dash-history-section">
          <h3>Post History</h3>
          <div class="history-list">
      `;

      brandData.posts.forEach((post, index) => {
        const totalEng = post.likes + post.comments + post.shares;
        const hasEngagement = totalEng > 0;
        const engClass = hasEngagement ? '' : 'no-engagement';

        html += `
          <div class="history-card ${engClass}">
            <div class="history-card-header">
              <span class="history-post-num">Post #${index + 1}</span>
              <span class="history-post-id">ID: ${post.post_id}</span>
            </div>
            <div class="history-topic">${post.topic}</div>
            <div class="history-content">${post.content.length > 200 ? post.content.substring(0, 200) + '...' : post.content}</div>
            ${post.hashtags && post.hashtags.length > 0 ? `
              <div class="history-hashtags">
                ${post.hashtags.map(h => `<span class="tag">#${h}</span>`).join('')}
              </div>
            ` : ''}
            <div class="history-engagement">
              <div class="history-stat ${post.likes > 0 ? 'has-data' : ''}">
                <span class="history-stat-icon">👍</span>
                <span>${post.likes}</span>
              </div>
              <div class="history-stat ${post.comments > 0 ? 'has-data' : ''}">
                <span class="history-stat-icon">💬</span>
                <span>${post.comments}</span>
              </div>
              <div class="history-stat ${post.shares > 0 ? 'has-data' : ''}">
                <span class="history-stat-icon">🔄</span>
                <span>${post.shares}</span>
              </div>
              <span class="history-date">${post.created_at ? new Date(post.created_at).toLocaleDateString() : ''}</span>
            </div>
          </div>
        `;
      });

      html += '</div></div>';
    } else {
      html += `
        <div class="dash-history-section">
          <h3>Post History</h3>
          <p class="hint">No posts yet. Head to the Generate Post tab to create your first post!</p>
        </div>
      `;
    }

    result.innerHTML = html;
    result.classList.remove('hidden');
    showToast(`Dashboard loaded for ${brandData.name}`);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(btn, false);
  }
});

// ── Brand Profile ────────────────────────────────────
document.getElementById('brand-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('.btn');
  const result = document.getElementById('brand-result');
  setLoading(btn, true);
  result.classList.add('hidden');

  try {
    const data = await apiCall('/brand', {
      name: document.getElementById('brand-name').value,
      role: document.getElementById('brand-role').value,
      industry: document.getElementById('brand-industry').value,
      goals: document.getElementById('brand-goals').value,
      preferred_tone: document.getElementById('brand-tone').value,
    });

    // Auto-fill profile ID across tabs
    setActiveProfile(data.id);

    result.innerHTML = `
      <h3>Brand Profile #${data.id}</h3>
      <div class="profile-header">
        <div class="profile-badge">${data.name ? data.name.charAt(0).toUpperCase() : '?'}</div>
        <div class="profile-info">
          <div class="profile-name">${data.name}</div>
          <div class="profile-role">${data.role} · ${data.industry}</div>
        </div>
      </div>
      ${renderField('Tone', data.tone || '—')}
      ${renderField('Positioning', data.positioning_summary || '—')}
      <div class="result-field">
        <div class="result-label">Content Themes</div>
        ${renderTags(data.content_themes)}
      </div>
      <div class="result-field">
        <div class="result-label">Do</div>
        ${renderTags(data.do_guidelines)}
      </div>
      <div class="result-field">
        <div class="result-label">Don't</div>
        ${renderTags(data.dont_guidelines)}
      </div>
      <div class="dash-autofill-notice">
        ✅ Profile ID #${data.id} auto-filled across all tabs.
      </div>
    `;
    result.classList.remove('hidden');
    showToast(`Brand profile #${data.id} created!`);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(btn, false);
  }
});

// ── Generate Post ────────────────────────────────────
document.getElementById('generate-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('.btn');
  const result = document.getElementById('generate-result');
  setLoading(btn, true);
  result.classList.add('hidden');

  try {
    const data = await apiCall('/generate', {
      user_id: parseInt(document.getElementById('gen-user-id').value),
      topic: document.getElementById('gen-topic').value,
    });

    result.innerHTML = `
      <div class="post-header">
        <h3>Post #${data.post_id}</h3>
        <button class="copy-btn-inline" onclick="copyPost(this)" title="Copy to clipboard">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
        </button>
      </div>
      <div class="post-preview">
        <div class="post-content">${data.post_content}</div>
      </div>
      <div class="result-field">
        <div class="result-label">Suggested Hashtags</div>
        ${renderTags(data.suggested_hashtags?.map(h => '#' + h))}
      </div>
    `;
    result.classList.remove('hidden');
    showToast(`Post #${data.post_id} generated!`);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(btn, false);
  }
});

// ── Engagement Predictor ─────────────────────────────
document.getElementById('predictor-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('.btn');
  const result = document.getElementById('predictor-result');
  setLoading(btn, true);
  result.classList.add('hidden');

  try {
    const data = await apiCall('/predict', {
      user_id: parseInt(document.getElementById('pred-user-id').value),
      draft_content: document.getElementById('pred-draft').value,
    });

    const score = data.overall_score;
    const scoreColor = score >= 70 ? 'score-high' : score >= 40 ? 'score-mid' : 'score-low';

    result.innerHTML = `
      <h3>Engagement Prediction</h3>
      <div class="predictor-top">
        <div class="score-badge ${scoreColor}">
          <div class="score-number">${score}</div>
          <div class="score-label">/ 100</div>
        </div>
        <div class="predicted-ranges">
          <div class="stat-card">
            <div class="stat-number">${data.predicted_likes}</div>
            <div class="stat-label">Predicted Likes</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${data.predicted_comments}</div>
            <div class="stat-label">Predicted Comments</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${data.predicted_shares}</div>
            <div class="stat-label">Predicted Shares</div>
          </div>
        </div>
      </div>

      <div class="rating-breakdown">
        <h4>Rating Breakdown</h4>
        ${renderRatingBar('Brand Alignment', data.brand_alignment)}
        ${renderRatingBar('Hook Strength', data.hook_strength)}
        ${renderRatingBar('Readability', data.readability)}
        ${renderRatingBar('Call-to-Action', data.call_to_action)}
      </div>

      <div class="feedback-section recommendation">
        <div class="feedback-icon">💡</div>
        <div>
          ${renderField('Improvement Tips', data.improvement_tips)}
        </div>
      </div>
    `;
    result.classList.remove('hidden');
    showToast(`Predicted score: ${score}/100`);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(btn, false);
  }
});

function renderRatingBar(label, value) {
  const color = value >= 70 ? '#34d399' : value >= 40 ? '#fbbf24' : '#f87171';
  return `
    <div class="rating-item">
      <div class="rating-header">
        <span class="rating-name">${label}</span>
        <span class="rating-value">${value}%</span>
      </div>
      <div class="rating-bar-bg">
        <div class="rating-bar-fill" style="width: ${value}%; background: ${color};"></div>
      </div>
    </div>
  `;
}

// ── Engagement ───────────────────────────────────────
document.getElementById('engagement-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('.btn');
  const result = document.getElementById('engagement-result');
  setLoading(btn, true);
  result.classList.add('hidden');

  try {
    const data = await apiCall('/engagement', {
      post_id: parseInt(document.getElementById('eng-post-id').value),
      likes: parseInt(document.getElementById('eng-likes').value),
      comments: parseInt(document.getElementById('eng-comments').value),
      shares: parseInt(document.getElementById('eng-shares').value),
    });

    result.innerHTML = `
      <h3>Engagement Logged ✓</h3>
      <div class="engagement-stats">
        <div class="stat-card">
          <div class="stat-number">${data.likes}</div>
          <div class="stat-label">Likes</div>
        </div>
        <div class="stat-card">
          <div class="stat-number">${data.comments}</div>
          <div class="stat-label">Comments</div>
        </div>
        <div class="stat-card">
          <div class="stat-number">${data.shares}</div>
          <div class="stat-label">Shares</div>
        </div>
      </div>
      ${renderField('Post ID', data.post_id)}
      ${renderField('Recorded At', new Date(data.created_at).toLocaleString())}
    `;
    result.classList.remove('hidden');
    showToast('Engagement logged!');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(btn, false);
  }
});

// ── Feedback ─────────────────────────────────────────
document.getElementById('feedback-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('.btn');
  const result = document.getElementById('feedback-result');
  setLoading(btn, true);
  result.classList.add('hidden');

  try {
    const data = await apiCall('/feedback', {
      user_id: parseInt(document.getElementById('fb-user-id').value),
    });

    result.innerHTML = `
      <h3>Strategy Feedback</h3>
      <div class="feedback-section">
        <div class="feedback-icon">📊</div>
        <div>
          ${renderField('Total Posts Analysed', data.total_posts)}
          ${renderField('Performance Summary', data.performance_summary)}
        </div>
      </div>
      <div class="feedback-section recommendation">
        <div class="feedback-icon">💡</div>
        <div>
          ${renderField('Recommendation', data.improvement_recommendation)}
        </div>
      </div>
    `;
    result.classList.remove('hidden');
    showToast('Feedback generated!');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(btn, false);
  }
});