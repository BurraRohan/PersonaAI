/* PersonaAI – Frontend Logic v3 (Dashboard Edition) */

const API = '';
// -- Auth ------------------------------------------------------
// The API key is never stored in this file. The user is asked for it once and
// it is kept in localStorage, so it survives tab closes and browser restarts.
// Call resetApiKey() from the console to be asked again.
function getAuthToken() {
  let token = localStorage.getItem('personaai_key');
  if (!token) {
    token = window.prompt('Enter your PersonaAI API key:');
    if (token) localStorage.setItem('personaai_key', token.trim());
  }
  return token ? token.trim() : '';
}

function clearAuthToken() {
  localStorage.removeItem('personaai_key');
}

// Escape hatch: type resetApiKey() in the browser console to re-enter the key.
window.resetApiKey = function () {
  clearAuthToken();
  showToast('API key cleared. It will be requested on your next action.');
};

// -- HTML escaping ---------------------------------------------
// Model output is untrusted text. Escape it before it goes near innerHTML,
// otherwise a topic containing markup will execute in the browser.
function esc(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Escape first, then render newlines as line breaks.
function escMultiline(value) {
  return esc(value).replace(/\n/g, '<br>');
}

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
      'Authorization': `Bearer ${getAuthToken()}`
    },
  };
  if (body && method === 'POST') {
    options.body = JSON.stringify(body);
  }
  const res = await fetch(API + url, options);

  if (res.status === 401 || res.status === 403) {
    clearAuthToken();  // forget the bad key so the user is prompted again
    throw new Error('Invalid or missing API key. Try again.');
  }

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
  return data;
}

function renderField(label, value) {
  return `<div class="result-field">
    <div class="result-label">${esc(label)}</div>
    <div class="result-value">${escMultiline(value)}</div>
  </div>`;
}

function renderTags(items) {
  if (!items || !items.length) return '<span class="result-value">—</span>';
  return `<div class="tag-list">${items.map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div>`;
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
          <div class="profile-badge">${esc(brandData.name ? brandData.name.charAt(0).toUpperCase() : '?')}</div>
          <div class="profile-info">
            <div class="profile-name">${esc(brandData.name)}</div>
            <div class="profile-role">${esc(brandData.role)} · ${esc(brandData.industry)}</div>
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
              ${post.status ? `<span class="status-badge status-${esc(post.status)}">${esc(post.status)}</span>` : ''}
            </div>
            <div class="history-topic">${esc(post.topic)}</div>
            <div class="history-content">${esc(post.content.length > 200 ? post.content.substring(0, 200) + '...' : post.content)}</div>
            ${post.hashtags && post.hashtags.length > 0 ? `
              <div class="history-hashtags">
                ${post.hashtags.map(h => `<span class="tag">#${esc(h)}</span>`).join('')}
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
        <div class="profile-badge">${esc(data.name ? data.name.charAt(0).toUpperCase() : '?')}</div>
        <div class="profile-info">
          <div class="profile-name">${esc(data.name)}</div>
          <div class="profile-role">${esc(data.role)} · ${esc(data.industry)}</div>
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
        <div class="post-content">${escMultiline(data.post_content)}</div>
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
    const postIdRaw = document.getElementById('pred-post-id').value.trim();
    const draftRaw = document.getElementById('pred-draft').value.trim();

    if (!postIdRaw && !draftRaw) {
      throw new Error('Enter a Post ID or paste a draft to evaluate.');
    }

    const payload = { user_id: parseInt(document.getElementById('pred-user-id').value) };
    if (postIdRaw) {
      payload.post_id = parseInt(postIdRaw);
    } else {
      payload.draft_content = draftRaw;
    }

    const data = await apiCall('/predict', payload);

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
            <div class="stat-number">${esc(data.predicted_likes)}</div>
            <div class="stat-label">Predicted Likes</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${esc(data.predicted_comments)}</div>
            <div class="stat-label">Predicted Comments</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">${esc(data.predicted_shares)}</div>
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

      ${renderDecisionPanel(data)}
    `;
    result.classList.remove('hidden');
    wireDecisionButtons(data.post_id);
    showToast(`Score: ${score}/100`);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(btn, false);
  }
});

// ── Human-in-the-loop review ──────────────────────────────────
// A score on its own decides nothing. When a saved post was evaluated, the
// person approves or rejects it here, and that decision is what gates the
// rest of the pipeline.

const STATUS_LABEL = {
  pending: 'Awaiting review',
  approved: 'Approved',
  rejected: 'Rejected',
};

function renderDecisionPanel(data) {
  if (!data.post_id) {
    return `
      <div class="decision-panel note-only">
        Pasted text is scored but not saved, so there is nothing to approve.
        Enter a Post ID to evaluate a generated post.
      </div>`;
  }

  const status = data.status || 'pending';

  if (status !== 'pending') {
    return `
      <div class="decision-panel">
        <div class="status-row">
          <span class="status-badge status-${esc(status)}">${esc(STATUS_LABEL[status] || status)}</span>
          <span class="status-note">Post #${esc(data.post_id)} has already been reviewed.</span>
        </div>
      </div>`;
  }

  return `
    <div class="decision-panel" id="decision-panel">
      <div class="status-row">
        <span class="status-badge status-pending">${esc(STATUS_LABEL.pending)}</span>
        <span class="status-note">Post #${esc(data.post_id)} will not enter the pipeline until you decide.</span>
      </div>
      <div class="field full-width">
        <label for="review-note">Note <span class="label-note">(optional, kept with the decision)</span></label>
        <input type="text" id="review-note" placeholder="e.g. Too generic, off-brand" />
      </div>
      <div class="decision-actions">
        <button type="button" class="btn btn-approve" id="btn-approve">Approve</button>
        <button type="button" class="btn btn-reject" id="btn-reject">Reject</button>
      </div>
    </div>`;
}

function wireDecisionButtons(postId) {
  if (!postId) return;

  const approve = document.getElementById('btn-approve');
  const reject = document.getElementById('btn-reject');
  if (!approve || !reject) return;

  const decide = async (decision, btn) => {
    setLoading(btn, true);
    try {
      const noteEl = document.getElementById('review-note');
      const res = await apiCall(`/posts/${postId}/review`, {
        decision,
        note: noteEl && noteEl.value.trim() ? noteEl.value.trim() : null,
      });

      const panel = document.getElementById('decision-panel');
      if (panel) {
        panel.innerHTML = `
          <div class="status-row">
            <span class="status-badge status-${esc(res.status)}">${esc(STATUS_LABEL[res.status] || res.status)}</span>
            <span class="status-note">Post #${esc(res.post_id)} marked ${esc(res.status)}.</span>
          </div>
          ${res.review_note ? `<div class="status-note">Note: ${esc(res.review_note)}</div>` : ''}`;
      }

      showToast(`Post ${res.status}`);
      loadPending();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setLoading(btn, false);
    }
  };

  approve.addEventListener('click', () => decide('approve', approve));
  reject.addEventListener('click', () => decide('reject', reject));
}

async function loadPending() {
  const userIdRaw = document.getElementById('pred-user-id').value.trim();
  const bar = document.getElementById('pending-bar');
  const count = document.getElementById('pending-count');
  const list = document.getElementById('pending-list');
  if (!bar || !count || !list) return;

  if (!userIdRaw) {
    bar.classList.add('hidden');
    list.innerHTML = '';
    return;
  }

  try {
    const posts = await apiCall(`/posts/pending/${parseInt(userIdRaw)}`, null, 'GET');

    if (!posts.length) {
      count.textContent = 'Nothing awaiting review.';
      bar.classList.remove('hidden');
      list.innerHTML = '';
      return;
    }

    count.textContent = `${posts.length} post${posts.length === 1 ? '' : 's'} awaiting review`;
    bar.classList.remove('hidden');

    list.innerHTML = posts.map(p => `
      <div class="pending-item">
        <div class="pending-head">
          <span class="status-badge status-pending">#${esc(p.post_id)}</span>
          <strong>${esc(p.topic)}</strong>
        </div>
        <div class="pending-preview">${esc(p.content.slice(0, 160))}${p.content.length > 160 ? '…' : ''}</div>
        <button type="button" class="btn-ghost pending-pick" data-post="${esc(p.post_id)}">Evaluate this</button>
      </div>`).join('');

    list.querySelectorAll('.pending-pick').forEach(btn => {
      btn.addEventListener('click', () => {
        document.getElementById('pred-post-id').value = btn.dataset.post;
        document.getElementById('pred-draft').value = '';
        document.getElementById('predictor-form').requestSubmit();
      });
    });
  } catch (err) {
    bar.classList.add('hidden');
    list.innerHTML = '';
  }
}

document.getElementById('pending-refresh').addEventListener('click', loadPending);
document.getElementById('pred-user-id').addEventListener('change', loadPending);

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