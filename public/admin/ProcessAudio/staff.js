/**
 * Staff Dashboard Page JavaScript
 * Displays audio analysis statistics grouped by customer support staff
 */

// State
let staffData = [];

// DOM Elements
const elements = {
    loadingState: document.getElementById('loadingState'),
    emptyState: document.getElementById('emptyState'),
    mainContent: document.getElementById('mainContent'),
    totalStaff: document.getElementById('totalStaff'),
    totalCalls: document.getElementById('totalCalls'),
    avgCallsPerStaff: document.getElementById('avgCallsPerStaff'),
    mostCommonMood: document.getElementById('mostCommonMood'),
    staffGrid: document.getElementById('staffGrid'),
    refreshBtn: document.getElementById('refreshBtn')
};

// Mood colors for consistency
const MOOD_COLORS = {
    HAPPY: { bg: '#10b981', class: 'happy' },
    SAD: { bg: '#3b82f6', class: 'sad' },
    ANGRY: { bg: '#ef4444', class: 'angry' },
    NEUTRAL: { bg: '#94a3b8', class: 'neutral' },
    FEARFUL: { bg: '#f59e0b', class: 'fearful' },
    DISGUSTED: { bg: '#8b5cf6', class: 'disgusted' },
    SURPRISED: { bg: '#ec4899', class: 'surprised' }
};

/**
 * Initialize the page
 */
async function init() {
    // Initialize sidebar
    await initSidebar();

    // Set up event listeners
    elements.refreshBtn.addEventListener('click', loadStaffData);

    // Load data
    await loadStaffData();
}

/**
 * Load staff statistics from API
 */
async function loadStaffData() {
    showLoading();

    try {
        const response = await fetch('/api/audio/stats/by-staff');
        const data = await response.json();

        if (data.success && data.staff && data.staff.length > 0) {
            staffData = data.staff;
            renderContent();
        } else {
            showEmpty();
        }
    } catch (error) {
        console.error('Failed to load staff data:', error);
        showEmpty();
    }
}

/**
 * Show loading state
 */
function showLoading() {
    elements.loadingState.style.display = 'block';
    elements.emptyState.style.display = 'none';
    elements.mainContent.style.display = 'none';
}

/**
 * Show empty state
 */
function showEmpty() {
    elements.loadingState.style.display = 'none';
    elements.emptyState.style.display = 'block';
    elements.mainContent.style.display = 'none';
}

/**
 * Render the main content
 */
function renderContent() {
    elements.loadingState.style.display = 'none';
    elements.emptyState.style.display = 'none';
    elements.mainContent.style.display = 'block';

    // Calculate summary stats
    const totalStaff = staffData.length;
    const totalCalls = staffData.reduce((sum, staff) => sum + staff.total_calls, 0);
    const avgCalls = totalStaff > 0 ? Math.round(totalCalls / totalStaff) : 0;

    // Find most common mood across all staff
    const moodTotals = {};
    staffData.forEach(staff => {
        if (staff.mood_counts) {
            Object.entries(staff.mood_counts).forEach(([mood, count]) => {
                moodTotals[mood] = (moodTotals[mood] || 0) + count;
            });
        }
    });

    let mostCommonMood = '-';
    let maxCount = 0;
    Object.entries(moodTotals).forEach(([mood, count]) => {
        if (count > maxCount) {
            maxCount = count;
            mostCommonMood = mood;
        }
    });

    // Update summary cards
    elements.totalStaff.textContent = totalStaff;
    elements.totalCalls.textContent = totalCalls;
    elements.avgCallsPerStaff.textContent = avgCalls;
    elements.mostCommonMood.textContent = mostCommonMood;

    // Render staff cards
    renderStaffGrid();
}

/**
 * Render the staff grid
 */
function renderStaffGrid() {
    elements.staffGrid.innerHTML = staffData.map(staff => {
        const initials = getInitials(staff.staff_name);
        const moodCounts = staff.mood_counts || {};
        const totalCalls = staff.total_calls || 0;

        // Calculate mood percentages for the bar
        const moodOrder = ['HAPPY', 'NEUTRAL', 'SAD', 'ANGRY', 'FEARFUL', 'DISGUSTED', 'SURPRISED'];
        const moodBarSegments = moodOrder.map(mood => {
            const count = moodCounts[mood] || 0;
            const percent = totalCalls > 0 ? (count / totalCalls * 100) : 0;
            return { mood, percent };
        }).filter(s => s.percent > 0);

        return `
            <div class="staff-card" onclick="viewStaffDetails('${escapeHtml(staff.staff_name)}')">
                <div class="staff-header">
                    <div class="staff-avatar">${initials}</div>
                    <div class="staff-info">
                        <div class="staff-name">${escapeHtml(staff.staff_name || 'Unknown')}</div>
                        <div class="staff-total"><strong>${totalCalls}</strong> calls analyzed</div>
                    </div>
                </div>

                <div class="mood-breakdown">
                    ${renderMoodItem('HAPPY', moodCounts.HAPPY)}
                    ${renderMoodItem('SAD', moodCounts.SAD)}
                    ${renderMoodItem('ANGRY', moodCounts.ANGRY)}
                    ${renderMoodItem('NEUTRAL', moodCounts.NEUTRAL)}
                    ${renderMoodItem('FEARFUL', moodCounts.FEARFUL)}
                    ${renderMoodItem('SURPRISED', moodCounts.SURPRISED)}
                </div>

                <div class="mood-bar-container">
                    <div class="mood-bar">
                        ${moodBarSegments.map(s =>
                            `<div class="mood-bar-segment ${s.mood.toLowerCase()}" style="width: ${s.percent}%"></div>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Render a single mood item
 */
function renderMoodItem(mood, count) {
    const moodLower = mood.toLowerCase();
    return `
        <div class="mood-item">
            <div class="mood-indicator ${moodLower}"></div>
            <span class="mood-label">${mood}</span>
            <span class="mood-count">${count || 0}</span>
        </div>
    `;
}

/**
 * Get initials from a name
 */
function getInitials(name) {
    if (!name) return '?';
    const parts = name.trim().split(/\s+/);
    if (parts.length === 1) {
        return parts[0].charAt(0).toUpperCase();
    }
    return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
}

/**
 * View staff details (navigate to search page filtered by staff)
 */
function viewStaffDetails(staffName) {
    // Navigate to audio search page with staff filter
    window.location.href = `../audio/index.html?staff=${encodeURIComponent(staffName)}`;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);
