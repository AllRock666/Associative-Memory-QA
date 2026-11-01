# -*- coding: utf-8 -*-
"""
Full compilation script to build static/index.html:
- Inter typography + exact zip file design system
- 72px left navigation rail + 300px secondary drawer
- 72px top header bar with branding and actions
- 100% emoji-free SVG vector icons across the entire application
- Preserves all 6 workspaces, 94 element IDs, and 63 JavaScript functions.
"""

import re
import os

with open('scripts/rebuild_full_project_ui.py', 'r', encoding='utf-8') as f:
    pass

with open('static/original_project.html', 'r', encoding='utf-8') as f:
    orig_html = f.read()

# Extract the main script (the second/last inline script block)
script_matches = re.findall(r'<script>(.*?)</script>', orig_html, re.DOTALL)
if len(script_matches) < 2:
    raise ValueError("Could not find main script in original_project.html")

orig_script = script_matches[-1]

# Clean out all emojis from the original script
replacements = [
    (r'"🌙"', r'""'), (r"'🌙'", r"''"),
    (r'"☀️"', r'""'), (r"'☀️'", r"''"),
    (r'"⚠️"', r'""'), (r"'⚠️'", r"''"),
    (r'"❌"', r'""'), (r"'❌'", r"''"),
    (r'"✅"', r'""'), (r"'✅'", r"''"),
    (r'"ℹ️"', r'""'), (r"'ℹ️'", r"''"),
    (r'"💬"', r'""'), (r"'💬'", r"''"),
    (r'"🤖"', r'""'), (r"'🤖'", r"''"),
    (r'"📁"', r'""'), (r"'📁'", r"''"),
    (r'"📊"', r'""'), (r"'📊'", r"''"),
    (r'"🌐"', r'""'), (r"'🌐'", r"''"),
    (r'"🦙"', r'""'), (r"'🦙'", r"''"),
    (r'"⚡"', r'""'), (r"'⚡'", r"''"),
    (r'"🧠"', r'""'), (r"'🧠'", r"''"),
    (r'"💎"', r'""'), (r"'💎'", r"''"),
    (r'"🔌"', r'""'), (r"'🔌'", r"''"),
    (r'"🛡️"', r'""'), (r"'🛡️'", r"''"),
    (r'"🔒"', r'""'), (r"'🔒'", r"''"),
    (r'"👁️"', r'""'), (r"'👁️'", r"''"),
    (r'"✨"', r'""'), (r"'✨'", r"''"),
    (r'"⚖️"', r'""'), (r"'⚖️'", r"''"),
    (r'"🗑️"', r'""'), (r"'🗑️'", r"''"),
    (r'"💡"', r'""'), (r"'💡'", r"''"),
    (r'"📑"', r'""'), (r"'📑'", r"''"),
    (r'🌐 Web Search: ON', r'Web Search: ON'),
    (r'🌐 Web Search: OFF', r'Web Search: OFF'),
    (r'🤖 Model:', r'Model:'),
    (r'🛡️ Offline', r'Offline'),
    (r'🛡️ <strong>100% Private', r'<strong>100% Private'),
    (r'🔒 <strong>Ephemeral', r'<strong>Ephemeral'),
    ('🦙 Ollama (${modelName})', '<span class="size-1.5 rounded-full bg-emerald-500 mr-1 inline-block"></span>Ollama (${modelName})'),
    ('🌐 Web search augmentation enabled', 'Web search augmentation enabled'),
    ('🌐 Web search disabled', 'Web search disabled'),
    ('⚠️ No Local LLM (Click for Cloud Key)', '<span class="size-1.5 rounded-full bg-amber-500 mr-1 inline-block"></span>No Local LLM (Click for Cloud Key)'),
    ('### 👤 User', '### User'),
    ('### 🧠 Hopfield Assistant', '### Hopfield Assistant'),
    ('📖 Reader', 'Reader'),
    ('⚡ ${(lat.total_ms).toFixed(0)}ms', '${(lat.total_ms).toFixed(0)}ms'),
    ('🛡️ *Extracted via Grounded Fallback', '*Extracted via Grounded Fallback'),
    ('let icon = "📄";', "let icon = getSvgIcon('file', 14, 'text-slate-400');"),
    ("if (ext === 'pdf') icon = \"📕\";", "if (ext === 'pdf') icon = getSvgIcon('file', 14, 'text-rose-500');"),
    ("else if (ext === 'pptx' || ext === 'ppt') icon = \"📙\";", "else if (ext === 'pptx' || ext === 'ppt') icon = getSvgIcon('file', 14, 'text-amber-500');"),
    ("else if (ext === 'docx' || ext === 'doc') icon = \"📘\";", "else if (ext === 'docx' || ext === 'doc') icon = getSvgIcon('file', 14, 'text-blue-500');"),
    ('<div class="text-3xl mb-2">📄</div>', '<div class="mb-2 text-slate-400 flex justify-center">${getSvgIcon(\'file\', 32)}</div>'),
    ('🔄 Tap to reveal answer', 'Tap to reveal answer'),
    ('🔄 Tap to flip back', 'Tap to flip back'),
    ('<strong>✓ Correct!</strong>', '<strong>Correct!</strong>'),
    ('<strong>✕ Incorrect.', '<strong>Incorrect.'),
    ('✓ ', ''),
    ('✕ ', ''),
    ('icon = "✓";', 'icon = "";'),
    ('icon = "✕";', 'icon = "";'),
]

cleaned_script = orig_script
for old, new in replacements:
    cleaned_script = cleaned_script.replace(old, new)

# Final safety pass: remove any lingering emojis or pictographs from the script
cleaned_script = re.sub(r'[\u2600-\u27bf\U00010000-\U0010ffff\ufe00-\ufe0f]', '', cleaned_script)

# SVG Icons Helper injected at top of script
svg_helper = r'''
// --- Pure Open-Source SVG Icons (Zero Emojis) ---
function getSvgIcon(name, size = 16, extraClass = '') {
    const icons = {
        chat: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />',
        reader: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />',
        studio: '<path d="M12 3 9.8 8.8 4 11l5.8 2.2L12 19l2.2-5.8L20 11l-5.8-2.2L12 3Z" /><path d="m5 3 .7 1.8L7.5 5.5l-1.8.7L5 8l-.7-1.8-1.8-.7 1.8-.7L5 3Z" />',
        kg: '<circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" /><line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />',
        study: '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" /><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />',
        analytics: '<line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" />',
        home: '<path d="m3 11 9-8 9 8" /><path d="M5 10v10h14V10M9 20v-6h6v6" />',
        settings: '<circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-2.83 2.83-.06-.06A1.7 1.7 0 0 0 15 19.4a1.7 1.7 0 0 0-1 .6 1.7 1.7 0 0 0-.4 1V21h-4v-.09A1.7 1.7 0 0 0 8.6 19.4a1.7 1.7 0 0 0-1.88.34l-.06.06-2.83-2.83.06-.06A1.7 1.7 0 0 0 4.2 15a1.7 1.7 0 0 0-.6-1 1.7 1.7 0 0 0-1-.4H2.5v-4h.09A1.7 1.7 0 0 0 4.2 8.6a1.7 1.7 0 0 0-.34-1.88l-.06-.06 2.83-2.83.06.06A1.7 1.7 0 0 0 8.6 4.2a1.7 1.7 0 0 0 1-.6 1.7 1.7 0 0 0 .4-1V2.5h4v.09a1.7 1.7 0 0 0 1 1.61 1.7 1.7 0 0 0 1.88-.34l.06-.06 2.83 2.83-.06.06a1.7 1.7 0 0 0-.31 1.91 1.7 1.7 0 0 0 .6 1 1.7 1.7 0 0 0 1 .4h.09v4H21a1.7 1.7 0 0 0-1.6 1Z" />',
        search: '<circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" />',
        plus: '<path d="M12 5v14M5 12h14" />',
        trash: '<path d="M3 6h18M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />',
        upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />',
        globe: '<circle cx="12" cy="12" r="10" /><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" /><path d="M2 12h20" />',
        mic: '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8" />',
        send: '<line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />',
        check: '<path d="m5 12 4 4L19 6" />',
        copy: '<rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />',
        speaker: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" /><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />',
        sun: '<circle cx="12" cy="12" r="5" /><line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" /><line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" /><line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" /><line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />',
        moon: '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />',
        eye: '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" /><circle cx="12" cy="12" r="3" />',
        cpu: '<rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" />',
        zap: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />',
        brain: '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z" /><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z" />',
        gem: '<path d="M6 3h12l4 6-10 13L2 9Z" /><path d="M11 3 8 9l4 13 4-13-3-6" /><path d="M2 9h20" />',
        shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />',
        bell: '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" /><path d="M10 21h4" />',
        chevronLeft: '<path d="m15 18-6-6 6-6" />',
        chevronRight: '<path d="m9 18 6-6-6-6" />',
        file: '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><polyline points="13 2 13 9 20 9" />',
        sidebar: '<rect x="3" y="3" width="18" height="18" rx="2" ry="2" /><line x1="9" y1="3" x2="9" y2="21" />'
    };
    return `<svg aria-hidden="true" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" class="${extraClass}">${icons[name] || ''}</svg>`;
}
'''

# Update script to use SVG icons
cleaned_script = cleaned_script.replace("icon.innerText = isDark ? '☀️' : '🌙';", "icon.innerHTML = isDark ? getSvgIcon('sun', 16) : getSvgIcon('moon', 16);")
cleaned_script = cleaned_script.replace("icon.innerText = isDark ? '🌙' : '☀️';", "icon.innerHTML = isDark ? getSvgIcon('moon', 16) : getSvgIcon('sun', 16);")
cleaned_script = cleaned_script.replace("icon.innerText = '🌙';", "icon.innerHTML = getSvgIcon('moon', 16);")
cleaned_script = cleaned_script.replace("icon.innerText = '☀️';", "icon.innerHTML = getSvgIcon('sun', 16);")

# Update workspace switcher to toggle active rail styling
custom_workspace_switcher = r'''
function switchMainWorkspace(view) {
    const views = ['chat', 'reader', 'studio', 'kg', 'study', 'analytics'];
    views.forEach(v => {
        const btn = document.getElementById(`nav-btn-${v}`);
        const panel = document.getElementById(`view-${v}`);
        const activeBar = document.getElementById(`rail-active-${v}`);
        if (v === view) {
            if (panel) { panel.classList.remove('hidden'); panel.classList.add('flex'); }
            if (btn) {
                btn.classList.add('bg-[#eef1ff]', 'text-[#3049b8]');
                btn.classList.remove('text-[#292d36]');
            }
            if (activeBar) activeBar.classList.remove('hidden');
        } else {
            if (panel) { panel.classList.add('hidden'); panel.classList.remove('flex'); }
            if (btn) {
                btn.classList.remove('bg-[#eef1ff]', 'text-[#3049b8]');
                btn.classList.add('text-[#292d36]');
            }
            if (activeBar) activeBar.classList.add('hidden');
        }
    });

    if (view === 'kg' && !cyInstance) loadKnowledgeGraph();
    if (view === 'analytics') refreshMemoryStats();
    if (view === 'reader' && multimodalDocs.length === 0) loadMultimodalDocs();
}
'''

cleaned_script = re.sub(
    r'function switchMainWorkspace\(view\)\s*\{.*?\n\}',
    custom_workspace_switcher.strip(),
    cleaned_script,
    flags=re.DOTALL
)

# Update mindmap support
cleaned_script = cleaned_script.replace(
    "const code = data.code || data.content;",
    "const code = data.content || data.code;"
)

# Update updateLLMUI logic
cleaned_script = re.sub(
    r'function updateLLMUI\(\)\s*\{.*?\n        \}',
    custom_update_llmui.strip(),
    cleaned_script,
    flags=re.DOTALL
)

# Update checkStatus to update window.__backendLLM and call updateLLMUI
cleaned_script = cleaned_script.replace(
    "const llmPill = document.getElementById('llm-status-pill');",
    "window.__backendLLM = data.llm;\n                    updateLLMUI();\n                    const llmPill = document.getElementById('llm-status-pill');"
)


# Assemble final script
final_script = svg_helper + "\n" + cleaned_script

full_html = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hopfield Multimodal Intelligence & Visual Learning Platform</title>

    <!-- Typography: Inter + JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

    <!-- Tailwind CSS with Dark Mode support -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        primary: {
                            50: '#f0f3ff',
                            100: '#e0e7ff',
                            500: '#3049b8',
                            600: '#253ca3',
                            700: '#1d2f82',
                        }
                    },
                    fontFamily: {
                        sans: ['"Inter"', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
                        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace']
                    }
                }
            }
        };
    </script>

    <!-- Markdown & Visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>

    <style>
        body {
            font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f7f8fa;
            color: #17191f;
            min-width: 320px;
        }
        .dark body {
            background: #090d16;
            color: #f1f5f9;
        }

        button, select, input, textarea {
            outline: none;
        }
        button { cursor: pointer; }

        h1 {
            font-size: clamp(1.4rem, 1.8vw, 1.85rem);
            line-height: 1.15;
            letter-spacing: -0.03em;
            font-weight: 600;
        }
        h2 {
            font-size: 0.95rem;
            line-height: 1.35;
            font-weight: 600;
        }

        /* Clean High-Production UI Tokens from Figma Make */
        .panel {
            border: 1px solid #e1e3e9;
            border-radius: 8px;
            background: white;
            box-shadow: 0 1px 2px rgb(24 29 44 / 0.02);
        }
        .dark .panel {
            border-color: #1e2638;
            background: #0f1523;
            box-shadow: 0 1px 2px rgb(0 0 0 / 0.2);
        }

        .control {
            display: inline-flex;
            min-height: 38px;
            align-items: center;
            gap: 8px;
            border: 1px solid #d9dce4;
            border-radius: 6px;
            background: white;
            padding: 0 12px;
            color: #333642;
            font-size: 12px;
            font-weight: 500;
            white-space: nowrap;
            transition: all 150ms ease;
        }
        .dark .control {
            border-color: #263147;
            background: #111827;
            color: #e2e8f0;
        }
        .control:hover {
            border-color: #3049b8;
            color: #3049b8;
        }
        .dark .control:hover {
            border-color: #6377e8;
            color: #a5b4fc;
        }

        .primary-button {
            display: inline-flex;
            min-height: 38px;
            align-items: center;
            gap: 8px;
            border-radius: 6px;
            background: #334dbb;
            padding: 0 16px;
            color: white;
            font-size: 13px;
            font-weight: 500;
            transition: background 150ms ease;
        }
        .primary-button:hover {
            background: #263da3;
        }

        .icon-button {
            display: grid;
            width: 36px;
            height: 36px;
            place-items: center;
            border-radius: 8px;
            color: #5d6373;
            transition: all 150ms ease;
        }
        .dark .icon-button {
            color: #94a3b8;
        }
        .icon-button:hover {
            background: #f2f4f8;
            color: #17191f;
        }
        .dark .icon-button:hover {
            background: #1e2638;
            color: white;
        }

        .search {
            display: flex;
            min-height: 36px;
            align-items: center;
            gap: 8px;
            border: 1px solid #d9dce4;
            border-radius: 6px;
            padding: 0 10px;
            color: #737783;
            background: white;
            transition: border-color 150ms ease, box-shadow 150ms ease;
        }
        .dark .search {
            border-color: #263147;
            background: #0f1523;
            color: #94a3b8;
        }
        .search:focus-within {
            border-color: #3049b8;
            box-shadow: 0 0 0 2px rgba(48, 73, 184, 0.12);
        }
        .search input {
            min-width: 0;
            flex: 1;
            font-size: 12px;
            color: #252832;
            border: none;
            background: transparent;
        }
        .dark .search input {
            color: #f1f5f9;
        }

        .tab {
            margin-bottom: -1px;
            border-bottom: 2px solid transparent;
            padding: 8px 14px;
            color: #696d78;
            font-size: 12px;
            font-weight: 500;
            transition: all 150ms ease;
        }
        .dark .tab { color: #8892a4; }
        .tab.active {
            border-color: #3049b8;
            color: #3049b8;
            font-weight: 600;
        }
        .dark .tab.active {
            border-color: #6377e8;
            color: #818cf8;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
        .dark ::-webkit-scrollbar-thumb { background: #263147; }
        .dark ::-webkit-scrollbar-thumb:hover { background: #374461; }

        /* 3D Flip Flashcards */
        .flashcard-scene { perspective: 1000px; }
        .flashcard-inner {
            transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            transform-style: preserve-3d;
        }
        .flashcard-flipped { transform: rotateY(180deg); }
        .flashcard-front, .flashcard-back {
            backface-visibility: hidden;
            -webkit-backface-visibility: hidden;
        }
        .flashcard-back { transform: rotateY(180deg); }

        /* Spinner */
        .spinner {
            border: 2px solid #e2e8f0;
            border-top-color: #3049b8;
            border-radius: 50%;
            width: 14px;
            height: 14px;
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Markdown Body */
        .markdown-body { font-size: 13px; line-height: 1.6; color: inherit; }
        .markdown-body h1, .markdown-body h2, .markdown-body h3 { font-weight: 600; margin-top: 0.8rem; margin-bottom: 0.3rem; }
        .markdown-body h1 { font-size: 1.15rem; }
        .markdown-body h2 { font-size: 1.05rem; }
        .markdown-body p { margin-bottom: 0.5rem; }
        .markdown-body ul, .markdown-body ol { margin-left: 1.25rem; margin-bottom: 0.5rem; list-style-type: disc; }
        .markdown-body code { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; background: #eef1ff; color: #263da3; padding: 2px 5px; border-radius: 4px; }
        .dark .markdown-body code { background: rgba(99, 102, 241, 0.2); color: #c7d2fe; }
        .markdown-body pre { background: #181d28; color: #f8fafc; padding: 10px 14px; border-radius: 6px; overflow-x: auto; margin: 0.5rem 0; font-family: monospace; }
        .markdown-body pre code { background: transparent; color: inherit; padding: 0; }
        .markdown-body table { width: 100%; border-collapse: collapse; margin: 0.8rem 0; font-size: 12px; }
        .markdown-body th, .markdown-body td { border: 1px solid #e2e8f0; padding: 6px 10px; }
        .dark .markdown-body th, .dark .markdown-body td { border-color: #263147; }
        .markdown-body th { background: #f8fafc; font-weight: 600; }
        .dark .markdown-body th { background: #111827; }
    </style>
</head>
<body class="h-screen flex flex-col overflow-hidden">

    <!-- ======================================================== -->
    <!-- 1. TOP HEADER (72px fixed, left-[72px] right-0)          -->
    <!-- ======================================================== -->
    <header class="fixed left-[72px] right-0 top-0 z-20 flex h-[72px] items-center justify-between border-b border-[#e5e7ec] dark:border-slate-800 bg-white dark:bg-slate-900 px-6 lg:px-8 transition-colors">
        <div class="flex items-center gap-5">
            <!-- Brand Logo -->
            <div class="text-[22px] font-semibold tracking-[-1.5px] text-[#111218] dark:text-white cursor-pointer" onclick="switchMainWorkspace('chat')">
                <span class="text-[#e64351] font-bold">hop</span>field
                <span class="ml-1.5 align-top text-[9px] tracking-normal text-[#51545e] dark:text-slate-400 font-normal">
                    Multimodal Intelligence
                </span>
            </div>

            <span class="hidden h-7 w-px bg-[#e0e2e7] dark:bg-slate-800 sm:block"></span>

            <!-- Status Badge -->
            <span id="pipeline-badge" class="hidden rounded bg-[#344fc2] px-3 py-1 text-[11px] font-medium tracking-wide text-white lg:inline">
                5-LEVEL MEMORY READY
            </span>

            <!-- Model Status Pill -->
            <button
                id="llm-status-pill"
                onclick="openLLMModal()"
                class="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/60 hover:bg-amber-100 text-[11px] font-medium text-amber-800 dark:text-amber-300 transition"
                title="Click to configure Model Settings"
            >
                <span class="size-2 rounded-full bg-amber-500 animate-pulse"></span>
                <span id="llm-status-text">Select Model</span>
            </button>
        </div>

        <div class="flex items-center gap-2">
            <!-- Web Search Toggle Button (Pure SVG Globe) -->
            <button
                type="button"
                id="web-search-toggle-btn"
                onclick="toggleWebSearch()"
                class="control"
                title="Toggle Live Web Search Augmentation"
            >
                <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                    <path d="M2 12h20" />
                </svg>
                <span class="size-1.5 rounded-full bg-emerald-500 animate-pulse" id="web-search-dot"></span>
                <span id="web-search-label">Web Search: ON</span>
            </button>

            <!-- New Session Button (Pure SVG Plus) -->
            <button
                onclick="createNewChat()"
                class="primary-button"
                title="Start a new conversation"
            >
                <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 5v14M5 12h14" />
                </svg>
                <span>New Session</span>
            </button>

            <!-- Theme Toggle Button (Pure SVG Moon/Sun) -->
            <button
                id="theme-toggle-btn"
                onclick="toggleTheme()"
                class="icon-button"
                title="Toggle Light / Dark Mode"
            >
                <span id="theme-toggle-icon">
                    <svg aria-hidden="true" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                    </svg>
                </span>
            </button>

            <!-- User Avatar -->
            <div class="grid size-9 place-items-center rounded-full bg-[#e44250] text-[12px] font-semibold text-white shadow-xs">
                HAM
            </div>
        </div>
    </header>

    <!-- ======================================================== -->
    <!-- 2. LEFT FIXED 72px NAVIGATION RAIL                      -->
    <!-- ======================================================== -->
    <aside class="fixed inset-y-0 left-0 z-30 flex w-[72px] flex-col items-center border-r border-[#e5e7ec] dark:border-slate-800 bg-white dark:bg-slate-900 py-4">
        <!-- Home / Reset Button -->
        <button
            aria-label="Home"
            onclick="switchMainWorkspace('chat')"
            class="mb-6 grid size-10 place-items-center rounded-xl text-[#15171c] dark:text-white transition hover:bg-[#f2f4fb] dark:hover:bg-slate-800"
            title="Chat Workspace"
        >
            <svg aria-hidden="true" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="m3 11 9-8 9 8" />
                <path d="M5 10v10h14V10M9 20v-6h6v6" />
            </svg>
        </button>

        <!-- 6 Workspaces Nav Rail -->
        <nav class="flex flex-1 flex-col gap-2">
            <!-- 1. Chat & RAG -->
            <button
                id="nav-btn-chat"
                aria-label="Chat & RAG"
                title="Chat & RAG Workspace"
                onclick="switchMainWorkspace('chat')"
                class="relative grid size-11 place-items-center rounded-xl transition bg-[#eef1ff] text-[#3049b8]"
            >
                <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <span id="rail-active-chat" class="absolute -right-[15px] h-7 w-[3px] rounded-l bg-[#3049b8]"></span>
            </button>

            <!-- 2. Visual Document Reader -->
            <button
                id="nav-btn-reader"
                aria-label="Visual Reader"
                title="Visual Document Reader"
                onclick="switchMainWorkspace('reader')"
                class="relative grid size-11 place-items-center rounded-xl transition text-[#292d36] dark:text-slate-400 hover:bg-[#f5f6f8] dark:hover:bg-slate-800"
            >
                <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                </svg>
                <span id="rail-active-reader" class="hidden absolute -right-[15px] h-7 w-[3px] rounded-l bg-[#3049b8]"></span>
            </button>

            <!-- 3. AI Visual Studio -->
            <button
                id="nav-btn-studio"
                aria-label="AI Visual Studio"
                title="AI Visual Studio (Mermaid & Plotly)"
                onclick="switchMainWorkspace('studio')"
                class="relative grid size-11 place-items-center rounded-xl transition text-[#292d36] dark:text-slate-400 hover:bg-[#f5f6f8] dark:hover:bg-slate-800"
            >
                <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 3 9.8 8.8 4 11l5.8 2.2L12 19l2.2-5.8L20 11l-5.8-2.2L12 3Z" />
                    <path d="m5 3 .7 1.8L7.5 5.5l-1.8.7L5 8l-.7-1.8-1.8-.7 1.8-.7L5 3Z" />
                </svg>
                <span id="rail-active-studio" class="hidden absolute -right-[15px] h-7 w-[3px] rounded-l bg-[#3049b8]"></span>
            </button>

            <!-- 4. Knowledge Graph -->
            <button
                id="nav-btn-kg"
                aria-label="Knowledge Graph"
                title="Level 4 Knowledge Graph Explorer"
                onclick="switchMainWorkspace('kg')"
                class="relative grid size-11 place-items-center rounded-xl transition text-[#292d36] dark:text-slate-400 hover:bg-[#f5f6f8] dark:hover:bg-slate-800"
            >
                <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="18" cy="5" r="3" />
                    <circle cx="6" cy="12" r="3" />
                    <circle cx="18" cy="19" r="3" />
                    <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
                    <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
                </svg>
                <span id="rail-active-kg" class="hidden absolute -right-[15px] h-7 w-[3px] rounded-l bg-[#3049b8]"></span>
            </button>

            <!-- 5. Study Assistant -->
            <button
                id="nav-btn-study"
                aria-label="Study Assistant"
                title="Study Assistant (Flashcards & Assessment)"
                onclick="switchMainWorkspace('study')"
                class="relative grid size-11 place-items-center rounded-xl transition text-[#292d36] dark:text-slate-400 hover:bg-[#f5f6f8] dark:hover:bg-slate-800"
            >
                <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
                </svg>
                <span id="rail-active-study" class="hidden absolute -right-[15px] h-7 w-[3px] rounded-l bg-[#3049b8]"></span>
            </button>

            <!-- 6. Memory Analytics -->
            <button
                id="nav-btn-analytics"
                aria-label="Analytics"
                title="5-Level Memory Diagnostics & Benchmark"
                onclick="switchMainWorkspace('analytics')"
                class="relative grid size-11 place-items-center rounded-xl transition text-[#292d36] dark:text-slate-400 hover:bg-[#f5f6f8] dark:hover:bg-slate-800"
            >
                <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10" />
                    <line x1="12" y1="20" x2="12" y2="4" />
                    <line x1="6" y1="20" x2="6" y2="14" />
                </svg>
                <span id="rail-active-analytics" class="hidden absolute -right-[15px] h-7 w-[3px] rounded-l bg-[#3049b8]"></span>
            </button>
        </nav>

        <!-- Bottom Actions -->
        <div class="flex flex-col gap-2">
            <!-- Sidebar Drawer Toggle -->
            <button
                aria-label="Toggle Sidebar"
                onclick="toggleSidebarDrawer()"
                title="Toggle Conversations & Documents Sidebar"
                class="grid size-11 place-items-center rounded-xl text-[#292d36] dark:text-slate-400 hover:bg-[#f5f6f8] dark:hover:bg-slate-800 transition"
            >
                <svg aria-hidden="true" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <line x1="9" y1="3" x2="9" y2="21" />
                </svg>
            </button>

            <!-- Settings Button -->
            <button
                id="llm-settings-btn"
                aria-label="Settings"
                onclick="openLLMModal()"
                title="Model & System Settings"
                class="grid size-11 place-items-center rounded-xl text-[#292d36] dark:text-slate-400 hover:bg-[#f5f6f8] dark:hover:bg-slate-800 transition"
            >
                <svg aria-hidden="true" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="3" />
                    <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-2.83 2.83-.06-.06A1.7 1.7 0 0 0 15 19.4a1.7 1.7 0 0 0-1 .6 1.7 1.7 0 0 0-.4 1V21h-4v-.09A1.7 1.7 0 0 0 8.6 19.4a1.7 1.7 0 0 0-1.88.34l-.06.06-2.83-2.83.06-.06A1.7 1.7 0 0 0 4.2 15a1.7 1.7 0 0 0-.6-1 1.7 1.7 0 0 0-1-.4H2.5v-4h.09A1.7 1.7 0 0 0 4.2 8.6a1.7 1.7 0 0 0-.34-1.88l-.06-.06 2.83-2.83.06.06A1.7 1.7 0 0 0 8.6 4.2a1.7 1.7 0 0 0 1-.6 1.7 1.7 0 0 0 .4-1V2.5h4v.09a1.7 1.7 0 0 0 1 1.61 1.7 1.7 0 0 0 1.88-.34l.06-.06 2.83 2.83-.06.06a1.7 1.7 0 0 0-.31 1.91 1.7 1.7 0 0 0 .6 1 1.7 1.7 0 0 0 1 .4h.09v4H21a1.7 1.7 0 0 0-1.6 1Z" />
                </svg>
            </button>
        </div>
    </aside>

    <!-- ======================================================== -->
    <!-- 3. SECONDARY SIDEBAR DRAWER (300px)                      -->
    <!-- ======================================================== -->
    <div id="sidebar-drawer" class="fixed left-[72px] top-[72px] bottom-0 w-[300px] border-r border-[#e5e7ec] dark:border-slate-800 bg-white dark:bg-slate-900 z-10 flex flex-col transition-all duration-200 shadow-xs">
        <!-- Tab Headers -->
        <div class="flex border-b border-[#e5e7ec] dark:border-slate-800 px-3 pt-2 bg-[#fafbfc] dark:bg-slate-900/60">
            <button id="tab-btn-chats" onclick="switchSidebarTab('chats')" class="tab active flex-1 text-center">
                Chats
            </button>
            <button id="tab-btn-docs" onclick="switchSidebarTab('docs')" class="tab flex-1 text-center">
                Documents <span id="docs-count-badge" class="ml-1 rounded bg-[#eef1ff] dark:bg-slate-800 px-1.5 py-0.2 text-[10px] font-bold text-[#3049b8] dark:text-indigo-400">0</span>
            </button>
            <button id="tab-btn-settings" onclick="switchSidebarTab('settings')" class="tab flex-1 text-center">
                Tuning
            </button>
        </div>

        <!-- TAB CONTENT 1: CHATS -->
        <div id="tab-content-chats" class="flex-1 flex flex-col p-3 overflow-hidden">
            <label class="search mb-2.5">
                <svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="7" />
                    <path d="m20 20-4-4" />
                </svg>
                <input type="text" id="chat-search-input" oninput="filterChats()" placeholder="Search conversations..." />
            </label>
            <div id="chats-list" class="flex-1 overflow-y-auto space-y-1 pr-1 text-[12px]">
                <div class="text-center py-8 text-[#88909f]">Loading conversations...</div>
            </div>
        </div>

        <!-- TAB CONTENT 2: DOCUMENTS -->
        <div id="tab-content-docs" class="flex-1 flex flex-col p-3 overflow-hidden hidden">
            <!-- Drag & Drop Upload Zone -->
            <div
                id="drop-zone"
                onclick="document.getElementById('file-upload-input').click()"
                class="rounded-lg border-2 border-dashed border-[#d9dce4] dark:border-slate-700 p-4 text-center hover:border-[#3049b8] transition cursor-pointer bg-[#fafbfc] dark:bg-slate-800/40 mb-3"
            >
                <input type="file" id="file-upload-input" class="hidden" onchange="uploadFile(this)" accept=".pdf,.docx,.pptx,.xlsx,.csv,.txt,.md,.png,.jpg,.jpeg">
                <svg aria-hidden="true" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3049b8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" class="mx-auto mb-1.5">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
                </svg>
                <span class="text-[12px] font-semibold text-[#1a1f2c] dark:text-slate-200 block">Click or Drop Documents</span>
                <span class="text-[10px] text-[#737887] block mt-0.5">PDF, DOCX, PPTX, Images</span>
            </div>
            <div id="upload-status" class="text-[11px] text-[#3049b8] font-semibold mb-2 px-1 hidden animate-pulse"></div>

            <!-- Document Search -->
            <label class="search mb-2.5">
                <svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="7" />
                    <path d="m20 20-4-4" />
                </svg>
                <input type="text" id="doc-search-input" oninput="filterDocuments()" placeholder="Filter indexed files..." />
            </label>
            <div id="documents-list" class="flex-1 overflow-y-auto space-y-1.5 pr-1 text-[12px]">
                <div class="text-center py-6 text-[#88909f]">No documents uploaded.</div>
            </div>
        </div>

        <!-- TAB CONTENT 3: SETTINGS & TUNING -->
        <div id="tab-content-settings" class="flex-1 p-3 overflow-y-auto space-y-4 text-[12px] hidden">
            <!-- Sidebar Model Selector -->
            <div class="panel p-3 space-y-3">
                <div class="flex items-center justify-between">
                    <span class="text-[10px] font-bold text-[#737782] uppercase tracking-wider">Model &amp; Provider</span>
                    <span id="sb-llm-badge" class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-[#eef1ff] dark:bg-slate-800 text-[#3049b8]">Local Ollama</span>
                </div>
                <div>
                    <select id="sb-provider-select" onchange="onSidebarProviderSelectChange(this.value)" class="w-full border border-[#d9dce4] dark:border-slate-700 rounded-md p-1.5 bg-white dark:bg-slate-800 text-xs">
                        <option value="ollama" selected>Local Ollama (llama3.2)</option>
                        <option value="groq">Groq (Llama-3.3 70B)</option>
                        <option value="openai">OpenAI (GPT-4o mini)</option>
                        <option value="gemini">Google Gemini (1.5 Flash)</option>
                        <option value="custom">Custom Endpoint</option>
                        <option value="offline">Offline Grounded (100% Private)</option>
                    </select>
                </div>
                <div id="sb-dynamic-fields" class="space-y-2">
                    <div id="sb-api-key-group" class="hidden">
                        <label class="block text-[11px] font-medium text-[#4b5262] mb-0.5">API Key (Session Only)</label>
                        <input type="password" id="sb-api-key" placeholder="API Key..." class="w-full border border-[#d9dce4] dark:border-slate-700 rounded-md p-1.5 font-mono text-[11px]" />
                    </div>
                    <div id="sb-model-name-group">
                        <label class="block text-[11px] font-medium text-[#4b5262] mb-0.5">Model Name</label>
                        <input type="text" id="sb-model-name" value="llama3.2" class="w-full border border-[#d9dce4] dark:border-slate-700 rounded-md p-1.5 font-mono text-[11px]" />
                    </div>
                    <div id="sb-base-url-group">
                        <label class="block text-[11px] font-medium text-[#4b5262] mb-0.5">Base URL</label>
                        <input type="text" id="sb-base-url" value="http://localhost:11434/v1" class="w-full border border-[#d9dce4] dark:border-slate-700 rounded-md p-1.5 font-mono text-[11px]" />
                    </div>
                    <div id="sb-offline-info" class="hidden p-2 rounded bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 text-[10px] text-emerald-800">
                        <strong>100% Private Offline Mode:</strong> Zero external API calls.
                    </div>
                    <div class="flex gap-1.5 pt-1">
                        <button onclick="testCurrentModel()" class="control flex-1 justify-center py-1 text-[11px]">Test</button>
                        <button onclick="saveCurrentModelFromSidebar()" class="primary-button flex-1 justify-center py-1 text-[11px]">Save</button>
                        <button onclick="clearLLMSession()" class="control py-1 text-[11px] text-[#bd493b]">Reset</button>
                    </div>
                    <div id="sb-test-status" class="text-[10px] font-medium hidden"></div>
                </div>
            </div>

            <!-- Hopfield Attractor Tuning -->
            <div class="panel p-3 space-y-3">
                <span class="text-[10px] font-bold text-[#737782] uppercase tracking-wider block">Hopfield Dynamics</span>
                <div>
                    <div class="flex justify-between text-[11px] font-medium mb-1">
                        <span>Inverse Temperature (&beta;)</span>
                        <span id="beta-val" class="text-[#3049b8] font-mono font-bold">8.0</span>
                    </div>
                    <input type="range" id="hopfield-beta" min="0.5" max="20" step="0.5" value="8.0" class="w-full accent-[#3049b8]" oninput="document.getElementById('beta-val').innerText=this.value" />
                </div>
                <div>
                    <div class="flex justify-between text-[11px] font-medium mb-1">
                        <span>Top-K Attractors</span>
                        <span id="topk-val" class="text-[#3049b8] font-mono font-bold">5</span>
                    </div>
                    <input type="range" id="top-k" min="1" max="15" step="1" value="5" class="w-full accent-[#3049b8]" oninput="document.getElementById('topk-val').innerText=this.value" />
                </div>
                <div>
                    <label class="block text-[11px] font-medium mb-1">Answer Style</label>
                    <select id="answer-style-select" class="w-full border border-[#d9dce4] dark:border-slate-700 rounded-md p-1.5 text-xs">
                        <option value="conversational" selected>Conversational (Grounded)</option>
                        <option value="detailed">Exhaustive &amp; Academic</option>
                        <option value="executive">Executive Summary</option>
                        <option value="concise">Concise 2-Mark Exam Mode</option>
                    </select>
                </div>
            </div>
        </div>
    </div>

    <!-- ======================================================== -->
    <!-- 4. MAIN WORKSPACE CONTAINER (6 WORKSPACES)               -->
    <!-- ======================================================== -->
    <main id="main-content-area" class="fixed left-[372px] right-0 top-[72px] bottom-0 overflow-y-auto bg-[#f7f8fa] dark:bg-slate-950 p-6 lg:p-8 transition-all duration-200">
        <div class="mx-auto max-w-[1440px] h-full flex flex-col">

            <!-- ==================================================== -->
            <!-- WORKSPACE 1: CHAT & RAG                              -->
            <!-- ==================================================== -->
            <div id="view-chat" class="flex-1 flex flex-col overflow-hidden">
                <!-- Scrollable Messages Container -->
                <div id="chat-scroll-container" class="flex-1 overflow-y-auto px-2 py-4 pb-44 scroll-smooth">
                    <!-- Welcome Hero State -->
                    <div id="welcome-state" class="max-w-3xl mx-auto py-10 px-4 text-center space-y-6">
                        <div>
                            <span class="rounded bg-[#eef1ff] dark:bg-slate-800 text-[#3049b8] dark:text-indigo-400 border border-[#c9d4ff] dark:border-slate-700 px-3 py-1 text-[11px] font-semibold">
                                Modern Hopfield Associative Memory Engine
                            </span>
                            <h1 class="mt-3 text-2xl sm:text-3xl font-bold tracking-tight text-[#111319] dark:text-white">
                                Multimodal Research &amp; Synthesis Assistant
                            </h1>
                            <p class="text-xs sm:text-sm text-[#646874] dark:text-slate-400 max-w-xl mx-auto mt-2 leading-relaxed">
                                Continuous attractor relaxation minimizing thermodynamic energy landscapes. Upload documents, extract visual slides, and query with verified citations.
                            </p>
                        </div>

                        <!-- 4 High-Impact Quick Prompts (Pure SVG Vectors, Zero Emojis) -->
                        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left max-w-2xl mx-auto pt-2">
                            <button onclick="setQueryAndSearch('What is generative AI and how does it differ from traditional AI?')" class="panel p-4 hover:border-[#3049b8] transition group">
                                <div class="flex items-center gap-2 mb-1">
                                    <span class="p-1.5 rounded-lg bg-[#eef1ff] dark:bg-slate-800 text-[#3049b8]">
                                        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3 9.8 8.8 4 11l5.8 2.2L12 19l2.2-5.8L20 11l-5.8-2.2L12 3Z" /></svg>
                                    </span>
                                    <span class="font-semibold text-xs text-[#111319] dark:text-white group-hover:text-[#3049b8]">What is Generative AI?</span>
                                </div>
                                <span class="text-[11px] text-[#737782] block truncate">Definition, synthesis models, and autonomous capabilities</span>
                            </button>

                            <button onclick="setQueryAndSearch('Explain Yet Another Resource Negotiator (YARN) architecture and daemons')" class="panel p-4 hover:border-[#3049b8] transition group">
                                <div class="flex items-center gap-2 mb-1">
                                    <span class="p-1.5 rounded-lg bg-[#eef1ff] dark:bg-slate-800 text-[#3049b8]">
                                        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
                                    </span>
                                    <span class="font-semibold text-xs text-[#111319] dark:text-white group-hover:text-[#3049b8]">YARN Architecture</span>
                                </div>
                                <span class="text-[11px] text-[#737782] block truncate">ResourceManager, NodeManager, and ApplicationMaster</span>
                            </button>

                            <button onclick="setQueryAndSearch('How does continuous modern Hopfield memory minimize energy?')" class="panel p-4 hover:border-[#3049b8] transition group">
                                <div class="flex items-center gap-2 mb-1">
                                    <span class="p-1.5 rounded-lg bg-[#eef1ff] dark:bg-slate-800 text-[#3049b8]">
                                        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /></svg>
                                    </span>
                                    <span class="font-semibold text-xs text-[#111319] dark:text-white group-hover:text-[#3049b8]">Continuous Hopfield Memory</span>
                                </div>
                                <span class="text-[11px] text-[#737782] block truncate">Exponential storage capacity, softmax attention, and energy minima</span>
                            </button>

                            <button onclick="setQueryAndSearch('Summarize key findings and tables from the uploaded documents')" class="panel p-4 hover:border-[#3049b8] transition group">
                                <div class="flex items-center gap-2 mb-1">
                                    <span class="p-1.5 rounded-lg bg-[#eef1ff] dark:bg-slate-800 text-[#3049b8]">
                                        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><polyline points="13 2 13 9 20 9" /></svg>
                                    </span>
                                    <span class="font-semibold text-xs text-[#111319] dark:text-white group-hover:text-[#3049b8]">Document Synthesis</span>
                                </div>
                                <span class="text-[11px] text-[#737782] block truncate">Synthesize core patterns, numerical tables, and cross-references</span>
                            </button>
                        </div>
                    </div>

                    <!-- Live Messages Feed -->
                    <div id="chat-history" class="max-w-3xl mx-auto space-y-4 hidden"></div>
                </div>

                <!-- Floating Bottom Composer Bar -->
                <div class="fixed bottom-6 left-[396px] right-8 max-w-3xl mx-auto z-20">
                    <!-- Pill recommendation chips (Zero emojis) -->
                    <div id="composer-chips" class="flex items-center gap-2 mb-2 overflow-x-auto py-0.5">
                        <button onclick="setQueryAndSearch('What are the key concepts in this document?')" class="control text-[11px] py-1 bg-white/90 dark:bg-slate-900/90 shadow-2xs backdrop-blur-md">
                            Key Concepts
                        </button>
                        <button onclick="setQueryAndSearch('Compare the theoretical advantages vs traditional retrieval')" class="control text-[11px] py-1 bg-white/90 dark:bg-slate-900/90 shadow-2xs backdrop-blur-md">
                            Compare Advantages
                        </button>
                        <button onclick="setQueryAndSearch('Extract and format all tabular or quantitative data')" class="control text-[11px] py-1 bg-white/90 dark:bg-slate-900/90 shadow-2xs backdrop-blur-md">
                            Extract Tables
                        </button>
                        <button onclick="clearCurrentChatMessages()" class="control text-[11px] py-1 text-[#bd493b] hover:bg-rose-50 ml-auto bg-white/90 shadow-2xs">
                            <svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 6h18M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" /></svg>
                            Clear Chat
                        </button>
                    </div>

                    <!-- Input Box Card -->
                    <div class="panel p-2.5 shadow-xl bg-white/95 dark:bg-slate-900/95 backdrop-blur-md border border-[#d9dce4] dark:border-slate-700">
                        <div class="flex items-end gap-2">
                            <!-- Voice Input (Pure SVG Mic) -->
                            <button id="voice-btn" onclick="toggleVoiceInput()" class="icon-button" title="Click to dictate question">
                                <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8" />
                                </svg>
                            </button>

                            <!-- Expanding Textarea -->
                            <textarea
                                id="query-input"
                                rows="1"
                                placeholder="Ask questions grounded in documents and associative memory..."
                                class="flex-1 max-h-36 bg-transparent text-[13px] text-[#111319] dark:text-slate-100 placeholder-[#7a7e89] resize-none py-2 leading-relaxed"
                                onkeydown="handleQueryKeydown(event)"
                                oninput="autoResizeTextarea(this)"
                            ></textarea>

                            <!-- Send Button (Pure SVG Send) -->
                            <button id="send-btn" onclick="sendQuery()" class="primary-button p-2.5 min-h-[38px] justify-center">
                                <svg id="send-icon" aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                                    <line x1="22" y1="2" x2="11" y2="13" />
                                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                                </svg>
                                <div id="search-spinner" class="spinner hidden" style="width: 14px; height: 14px; border-width: 2px; border-top-color: white;"></div>
                            </button>
                        </div>

                        <!-- Composer Footer Info -->
                        <div class="flex items-center justify-between pt-2 px-2 border-t border-[#f0f2f5] dark:border-slate-800 text-[10px] text-[#737782]">
                            <div class="flex items-center gap-2">
                                <span class="hover:text-[#3049b8] cursor-pointer flex items-center gap-1 font-medium" onclick="openLLMModal()">
                                    <span>Model:</span>
                                    <strong id="composer-model-indicator" class="font-mono text-[#252832] dark:text-slate-300">Local Ollama (llama3.2)</strong>
                                </span>
                                <span>&bull;</span>
                                <span id="composer-chunks-count">Ready</span>
                            </div>
                            <div class="hidden sm:block">
                                <kbd class="px-1.5 py-0.5 rounded bg-[#edf0f5] dark:bg-slate-800 text-[9px] font-mono">↵ Send</kbd>
                                <span class="mx-1">or</span>
                                <kbd class="px-1.5 py-0.5 rounded bg-[#edf0f5] dark:bg-slate-800 text-[9px] font-mono">Shift+↵ New Line</kbd>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ==================================================== -->
            <!-- WORKSPACE 2: VISUAL DOCUMENT READER                  -->
            <!-- ==================================================== -->
            <div id="view-reader" class="flex-1 flex flex-col overflow-hidden hidden">
                <div class="flex items-center justify-between pb-3 border-b border-[#e5e7ec] dark:border-slate-800 mb-4">
                    <div class="flex items-center gap-3">
                        <select id="reader-doc-select" onchange="loadReaderDocument(this.value)" class="control text-xs">
                            <option value="">Select Document...</option>
                        </select>
                        <div class="flex items-center gap-1">
                            <button onclick="prevReaderPage()" class="control px-2 py-1 text-xs">
                                <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m15 18-6-6 6-6" /></svg>
                            </button>
                            <span id="reader-page-indicator" class="text-xs font-mono font-bold px-2 text-[#2d313d] dark:text-slate-300">Page 1</span>
                            <button onclick="nextReaderPage()" class="control px-2 py-1 text-xs">
                                <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6" /></svg>
                            </button>
                        </div>
                    </div>
                    <button onclick="switchMainWorkspace('chat')" class="text-xs text-[#3049b8] font-medium hover:underline">
                        ← Back to Chat
                    </button>
                </div>

                <div class="flex-1 flex gap-4 overflow-hidden">
                    <div class="w-1/2 panel flex items-center justify-center p-4 overflow-auto bg-[#fafbfc] dark:bg-slate-900/50">
                        <div id="reader-image-wrap" class="max-h-full flex items-center justify-center">
                            <div class="text-center p-8 text-[#898e9a] text-xs">
                                Select a document above to inspect its visual slides and extracted diagrams.
                            </div>
                        </div>
                    </div>
                    <div class="w-1/2 panel p-4 flex flex-col overflow-hidden">
                        <h2 class="text-xs font-bold text-[#454955] dark:text-slate-300 uppercase tracking-wider mb-2">Multimodal Page Intelligence</h2>
                        <div id="reader-elements-container" class="flex-1 overflow-y-auto space-y-3 pr-1 text-xs text-[#2b2f3b] dark:text-slate-300">
                            <div class="text-[#898e9a] text-center py-8">Select a document to inspect structural elements and tables.</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ==================================================== -->
            <!-- WORKSPACE 3: AI VISUAL STUDIO                        -->
            <!-- ==================================================== -->
            <div id="view-studio" class="flex-1 flex flex-col overflow-hidden hidden">
                <div class="flex items-center justify-between pb-3 border-b border-[#e5e7ec] dark:border-slate-800 mb-4">
                    <div>
                        <h1>AI Visual Studio</h1>
                        <p class="text-[12px] text-[#737782]">Generate grounded diagrams, mind maps, and interactive numerical charts</p>
                    </div>
                    <div class="flex items-center gap-2">
                        <select id="studio-type-select" class="control text-xs">
                            <option value="mindmap">Mind Map (Mermaid)</option>
                            <option value="flowchart">Workflow Flowchart (Mermaid)</option>
                            <option value="architecture">System Architecture (Mermaid)</option>
                            <option value="sequence">Sequence Diagram (Mermaid)</option>
                            <option value="roadmap">Study Roadmap (Mermaid)</option>
                            <option value="bar">Bar Chart (Plotly)</option>
                            <option value="line">Line Trend (Plotly)</option>
                            <option value="pie">Distribution Pie (Plotly)</option>
                        </select>
                        <button onclick="generateStudioViz()" class="primary-button text-xs">
                            Generate
                        </button>
                    </div>
                </div>

                <div class="mb-3">
                    <label class="search">
                        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></svg>
                        <input type="text" id="studio-prompt-input" placeholder="e.g. Generate a mind map of operating systems or bar chart comparing workloads..." />
                    </label>
                </div>

                <div class="flex-1 panel p-4 flex flex-col overflow-auto bg-[#fafbfc] dark:bg-slate-900/50">
                    <div id="studio-loading" class="hidden text-center py-12 text-xs text-[#3049b8] font-semibold animate-pulse">Generating verified visualization...</div>
                    <div id="studio-notice" class="hidden mb-3 p-3 rounded bg-amber-50 border border-amber-200 text-xs text-amber-800"></div>
                    <div id="studio-render-container" class="flex-1 flex items-center justify-center overflow-auto text-xs text-[#898e9a]">
                        Enter a prompt above and click Generate to build diagrams or interactive charts.
                    </div>
                </div>
            </div>

            <!-- ==================================================== -->
            <!-- WORKSPACE 4: KNOWLEDGE GRAPH EXPLORER                -->
            <!-- ==================================================== -->
            <div id="view-kg" class="flex-1 flex flex-col overflow-hidden hidden">
                <div class="flex items-center justify-between pb-3 border-b border-[#e5e7ec] dark:border-slate-800 mb-4">
                    <div>
                        <h1>Knowledge Graph Explorer</h1>
                        <p class="text-[12px] text-[#737782]">Level 4 Multi-Relational Knowledge Network grounded across documents and pages</p>
                    </div>
                    <div class="flex items-center gap-2">
                        <label class="search w-52">
                            <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></svg>
                            <input type="text" id="kg-search-input" oninput="filterKGNodes(this.value)" placeholder="Search node in graph..." />
                        </label>
                        <select id="kg-layout-select" onchange="changeKGLayout(this.value)" class="control text-xs">
                            <option value="cose">Force-Directed (Cose)</option>
                            <option value="concentric">Concentric</option>
                            <option value="circle">Circle</option>
                            <option value="breadthfirst">Hierarchical Tree</option>
                        </select>
                        <button onclick="loadKnowledgeGraph()" class="control text-xs">Refresh</button>
                    </div>
                </div>

                <div class="flex-1 flex gap-4 overflow-hidden">
                    <div id="cy-container" class="flex-1 panel bg-slate-900 border border-slate-800"></div>
                    <div class="w-72 panel p-3 flex flex-col text-xs">
                        <h2 class="text-xs font-bold text-[#454955] dark:text-slate-300 uppercase tracking-wider mb-2">Node Inspector</h2>
                        <div id="kg-node-details" class="text-[#737782] leading-relaxed">
                            Click any entity or document node in the canvas to inspect relations and source provenance.
                        </div>
                    </div>
                </div>
            </div>

            <!-- ==================================================== -->
            <!-- WORKSPACE 5: STUDY ASSISTANT                         -->
            <!-- ==================================================== -->
            <div id="view-study" class="flex-1 flex flex-col overflow-hidden hidden">
                <div class="flex items-center justify-between pb-3 border-b border-[#e5e7ec] dark:border-slate-800 mb-4">
                    <div>
                        <h1>AI Study Workspace</h1>
                        <p class="text-[12px] text-[#737782]">Active recall flashcards and pedagogical multiple-choice assessments</p>
                    </div>
                    <div class="flex items-center gap-2">
                        <label class="search w-64">
                            <input type="text" id="study-topic-input" placeholder="Topic or Document Subject..." />
                        </label>
                        <button onclick="loadStudyDeck()" class="primary-button text-xs">
                            Build Study Deck
                        </button>
                    </div>
                </div>

                <div class="flex-1 flex gap-4 overflow-y-auto">
                    <!-- Left: Flashcards Deck with 3D Flip -->
                    <div class="w-1/2 flex flex-col space-y-3">
                        <div class="flex items-center justify-between">
                            <h2 class="text-xs font-bold uppercase tracking-wider text-[#454955] dark:text-slate-300">Active Recall Flashcards</h2>
                            <span id="flashcard-count-indicator" class="text-[10px] text-[#737782]">Click card to flip</span>
                        </div>
                        <div id="study-flashcards-list" class="space-y-3 overflow-y-auto pr-1">
                            <div class="text-xs text-[#898e9a] text-center py-8">Generate a study deck above to drill flashcards.</div>
                        </div>
                    </div>

                    <!-- Right: MCQ Assessment -->
                    <div class="w-1/2 flex flex-col space-y-3 border-l border-[#e5e7ec] dark:border-slate-800 pl-4">
                        <div class="flex items-center justify-between">
                            <h2 class="text-xs font-bold uppercase tracking-wider text-[#454955] dark:text-slate-300">Assessment Quiz</h2>
                            <span id="quiz-score-badge" class="control py-0.5 px-2 text-[10px] font-bold">Score: 0 / 0</span>
                        </div>
                        <div id="study-mcqs-list" class="space-y-4 overflow-y-auto pr-1">
                            <div class="text-xs text-[#898e9a] text-center py-8">Questions and distractor rationales will appear here.</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ==================================================== -->
            <!-- WORKSPACE 6: MEMORY ANALYTICS & BENCHMARK            -->
            <!-- ==================================================== -->
            <div id="view-analytics" class="flex-1 flex flex-col overflow-y-auto hidden space-y-4">
                <div class="flex items-center justify-between pb-3 border-b border-[#e5e7ec] dark:border-slate-800">
                    <div>
                        <h1>5-Level Memory Diagnostics &amp; Benchmark</h1>
                        <p class="text-[12px] text-[#737782]">Thermodynamic energy landscape monitoring and standard retrieval benchmarks</p>
                    </div>
                    <button onclick="runLiveBenchmark()" class="primary-button text-xs">
                        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                        <span>Run Live Benchmark</span>
                    </button>
                </div>

                <!-- 5-Level Memory Hierarchy Cards -->
                <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
                    <article class="panel p-4">
                        <span class="text-[10px] uppercase font-bold text-[#737782] block">Level 1: Raw Chunks</span>
                        <div id="stat-l1" class="text-2xl font-bold text-[#111319] dark:text-white mt-1">0</div>
                    </article>
                    <article class="panel p-4">
                        <span class="text-[10px] uppercase font-bold text-[#3049b8] block">Level 2: Embeddings</span>
                        <div id="stat-l2" class="text-2xl font-bold text-[#3049b8] mt-1">0</div>
                    </article>
                    <article class="panel p-4">
                        <span class="text-[10px] uppercase font-bold text-[#5c35b8] block">Level 3: Hopfield &beta;</span>
                        <div id="stat-l3" class="text-2xl font-bold text-[#5c35b8] mt-1">8.0</div>
                    </article>
                    <article class="panel p-4">
                        <span class="text-[10px] uppercase font-bold text-[#2c8069] block">Level 4: KG Nodes</span>
                        <div id="stat-l4" class="text-2xl font-bold text-[#2c8069] mt-1">0</div>
                    </article>
                    <article class="panel p-4">
                        <span class="text-[10px] uppercase font-bold text-[#9d3278] block">Level 5: Episodic</span>
                        <div id="stat-l5" class="text-2xl font-bold text-[#9d3278] mt-1">0</div>
                    </article>
                </div>

                <!-- Live Benchmark Output Viewport -->
                <div class="panel p-4 bg-[#131924] text-white border-slate-800 font-mono text-xs shadow-md">
                    <h2 class="text-emerald-400 font-bold mb-2 flex items-center gap-2">
                        <span class="size-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span>Hopfield Convergence &amp; Standard Retrieval Benchmark:</span>
                    </h2>
                    <div id="benchmark-output" class="leading-relaxed text-slate-300">
                        Click "Run Live Benchmark" above to evaluate Recall@1, Recall@5, NDCG@5, Mean Latency, and Attractor Energy Basin stability.
                    </div>
                </div>

                <!-- Plotly Chart Container -->
                <div class="panel p-4">
                    <h2 class="text-xs font-bold text-[#454955] dark:text-slate-300 uppercase tracking-wider mb-2">Thermodynamic Energy Landscape Relaxation Curve</h2>
                    <div id="energy-chart-container" class="w-full h-64 flex items-center justify-center text-xs text-[#898e9a]">
                        Run a search query or benchmark to visualize the Hopfield energy minimization trajectory.
                    </div>
                </div>
            </div>

        </div>
    </main>

    <!-- ======================================================== -->
    <!-- 5. MODEL SETTINGS & INFERENCE MODAL                      -->
    <!-- ======================================================== -->
    <div id="llm-config-modal" class="fixed inset-0 z-50 grid place-items-center bg-[#11141f]/45 p-4 hidden" onclick="closeLLMModal()">
        <div class="w-full max-w-xl overflow-hidden rounded-xl bg-white dark:bg-slate-900 shadow-2xl border border-[#e2e4ea] dark:border-slate-800" onclick="event.stopPropagation()">
            <!-- Modal Header -->
            <div class="flex items-center justify-between border-b border-[#e4e6eb] dark:border-slate-800 px-6 py-4">
                <div>
                    <h2 class="text-[16px] font-semibold text-[#111319] dark:text-white">Model &amp; Inference Settings</h2>
                    <p class="text-[12px] text-[#737782] dark:text-slate-400">Select model provider, configure API credentials, and tune parameters</p>
                </div>
                <button onclick="closeLLMModal()" class="icon-button">
                    <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M18 6 6 18M6 6l12 12" /></svg>
                </button>
            </div>

            <!-- Modal Content -->
            <div class="p-6 space-y-4 max-h-[75vh] overflow-y-auto text-[13px]">
                <!-- Provider Cards (6 options, Pure SVG Icons, Zero Emojis) -->
                <div>
                    <label class="block font-medium text-[#2d313d] dark:text-slate-200 mb-2">Choose Inference Provider</label>
                    <div class="grid grid-cols-3 gap-2" id="modal-provider-cards">
                        <button type="button" onclick="selectModalProvider('ollama')" id="card-prov-ollama" class="p-2.5 rounded-lg border-2 text-left transition border-[#3049b8] bg-[#eef1ff] dark:bg-slate-800">
                            <div class="font-medium text-[#11141c] dark:text-white flex items-center justify-between">
                                <span class="flex items-center gap-1.5">
                                    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /></svg>
                                    Ollama
                                </span>
                                <span class="size-2 rounded-full bg-[#3049b8]" id="dot-prov-ollama"></span>
                            </div>
                            <span class="text-[10px] text-[#636876] dark:text-slate-400 block mt-0.5">Local CPU/GPU</span>
                        </button>

                        <button type="button" onclick="selectModalProvider('groq')" id="card-prov-groq" class="p-2.5 rounded-lg border-2 text-left transition border-[#e1e3e9] dark:border-slate-700 hover:border-[#b8c2ec]">
                            <div class="font-medium text-[#11141c] dark:text-white flex items-center justify-between">
                                <span class="flex items-center gap-1.5">
                                    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
                                    Groq
                                </span>
                                <span class="size-2 rounded-full bg-slate-300" id="dot-prov-groq"></span>
                            </div>
                            <span class="text-[10px] text-[#636876] dark:text-slate-400 block mt-0.5">Llama-3.3 70B</span>
                        </button>

                        <button type="button" onclick="selectModalProvider('openai')" id="card-prov-openai" class="p-2.5 rounded-lg border-2 text-left transition border-[#e1e3e9] dark:border-slate-700 hover:border-[#b8c2ec]">
                            <div class="font-medium text-[#11141c] dark:text-white flex items-center justify-between">
                                <span class="flex items-center gap-1.5">
                                    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z" /></svg>
                                    OpenAI
                                </span>
                                <span class="size-2 rounded-full bg-slate-300" id="dot-prov-openai"></span>
                            </div>
                            <span class="text-[10px] text-[#636876] dark:text-slate-400 block mt-0.5">GPT-4o mini</span>
                        </button>

                        <button type="button" onclick="selectModalProvider('gemini')" id="card-prov-gemini" class="p-2.5 rounded-lg border-2 text-left transition border-[#e1e3e9] dark:border-slate-700 hover:border-[#b8c2ec]">
                            <div class="font-medium text-[#11141c] dark:text-white flex items-center justify-between">
                                <span class="flex items-center gap-1.5">
                                    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6 3h12l4 6-10 13L2 9Z" /></svg>
                                    Gemini
                                </span>
                                <span class="size-2 rounded-full bg-slate-300" id="dot-prov-gemini"></span>
                            </div>
                            <span class="text-[10px] text-[#636876] dark:text-slate-400 block mt-0.5">1.5 Flash</span>
                        </button>

                        <button type="button" onclick="selectModalProvider('custom')" id="card-prov-custom" class="p-2.5 rounded-lg border-2 text-left transition border-[#e1e3e9] dark:border-slate-700 hover:border-[#b8c2ec]">
                            <div class="font-medium text-[#11141c] dark:text-white flex items-center justify-between">
                                <span class="flex items-center gap-1.5">
                                    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88" /></svg>
                                    Custom
                                </span>
                                <span class="size-2 rounded-full bg-slate-300" id="dot-prov-custom"></span>
                            </div>
                            <span class="text-[10px] text-[#636876] dark:text-slate-400 block mt-0.5">OpenAI URL</span>
                        </button>

                        <button type="button" onclick="selectModalProvider('offline')" id="card-prov-offline" class="p-2.5 rounded-lg border-2 text-left transition border-[#e1e3e9] dark:border-slate-700 hover:border-[#b8c2ec]">
                            <div class="font-medium text-[#11141c] dark:text-white flex items-center justify-between">
                                <span class="flex items-center gap-1.5">
                                    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                                    Offline
                                </span>
                                <span class="size-2 rounded-full bg-slate-300" id="dot-prov-offline"></span>
                            </div>
                            <span class="text-[10px] text-[#636876] dark:text-slate-400 block mt-0.5">100% Private</span>
                        </button>
                    </div>
                </div>

                <!-- Provider Fields -->
                <div id="modal-provider-fields" class="space-y-3">
                    <div id="modal-api-key-container" class="hidden">
                        <label class="block font-medium text-[#2d313d] dark:text-slate-200 mb-1">API Key (Session Storage Only)</label>
                        <div class="relative">
                            <input type="password" id="modal-api-key" placeholder="Enter API Key..." class="w-full border border-[#d9dce4] dark:border-slate-700 rounded-md p-2 pr-9 font-mono text-[12px] bg-white dark:bg-slate-800" />
                            <button type="button" onclick="togglePasswordVisibility('modal-api-key')" class="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600">
                                <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" /><circle cx="12" cy="12" r="3" /></svg>
                            </button>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-3" id="modal-model-url-grid">
                        <div id="modal-model-group">
                            <label class="block font-medium text-[#2d313d] dark:text-slate-200 mb-1">Model Name</label>
                            <input type="text" id="modal-model-name" value="llama3.2" class="w-full border border-[#d9dce4] dark:border-slate-700 rounded-md p-2 font-mono text-[12px] bg-white dark:bg-slate-800" />
                        </div>
                        <div id="modal-url-group">
                            <label class="block font-medium text-[#2d313d] dark:text-slate-200 mb-1">Base URL</label>
                            <input type="text" id="modal-base-url" value="http://localhost:11434/v1" class="w-full border border-[#d9dce4] dark:border-slate-700 rounded-md p-2 font-mono text-[12px] bg-white dark:bg-slate-800" />
                        </div>
                    </div>

                    <div id="modal-offline-notice" class="hidden rounded-lg bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900 p-3 text-[12px] text-emerald-800 dark:text-emerald-300">
                        <strong>Offline Grounded Mode:</strong> No outbound network traffic or external API calls. Extracted answers and mind maps are derived deterministically from Hopfield attractor dynamics and indexed evidence.
                    </div>

                    <div id="modal-session-security-notice" class="rounded-lg bg-[#f0f3ff] dark:bg-slate-800 border border-[#d4deff] dark:border-slate-700 p-3 text-[12px] text-[#24357c] dark:text-slate-300 flex items-start gap-2">
                        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="flex-shrink-0 mt-0.5"><rect x="3" y="11" width="18" height="11" rx="2" ry="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></svg>
                        <span><strong>Session Memory:</strong> Credentials remain strictly in browser <code class="px-1 py-0.2 rounded bg-white dark:bg-slate-900 border text-[11px]">sessionStorage</code> and are never written to disk or logs.</span>
                    </div>
                </div>

                <!-- Sliders: Temperature & Tokens -->
                <div class="pt-3 border-t border-[#eceef2] dark:border-slate-800 space-y-3">
                    <span class="text-[11px] font-semibold uppercase tracking-wider text-[#737782] block">Hyperparameter Tuning</span>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <div class="flex justify-between text-[12px] font-medium mb-1">
                                <span>Temperature</span>
                                <span id="modal-temp-val" class="text-[#3049b8] font-mono font-bold">0.2</span>
                            </div>
                            <input type="range" id="modal-temp-slider" min="0.0" max="1.0" step="0.05" value="0.2" class="w-full accent-[#3049b8]" oninput="document.getElementById('modal-temp-val').innerText=this.value" />
                        </div>
                        <div>
                            <div class="flex justify-between text-[12px] font-medium mb-1">
                                <span>Max Tokens</span>
                                <span id="modal-tokens-val" class="text-[#3049b8] font-mono font-bold">2048</span>
                            </div>
                            <input type="range" id="modal-tokens-slider" min="256" max="8192" step="256" value="2048" class="w-full accent-[#3049b8]" oninput="document.getElementById('modal-tokens-val').innerText=this.value" />
                        </div>
                    </div>
                </div>

                <!-- Status Feedback Box -->
                <div id="modal-status-box" class="hidden p-3 rounded-lg border text-[12px] font-medium"></div>
            </div>

            <!-- Footer Actions -->
            <div class="flex items-center justify-between border-t border-[#e4e6eb] dark:border-slate-800 px-6 py-4 bg-[#fafbfc] dark:bg-slate-900/50">
                <button onclick="clearLLMSession()" class="text-[12px] text-[#bd493b] hover:underline font-medium">
                    Reset to Local Default
                </button>
                <div class="flex gap-2">
                    <button onclick="testModalModel()" class="control">
                        <span id="modal-test-spinner" class="spinner hidden mr-1"></span>
                        Test Connection
                    </button>
                    <button onclick="saveModalModelSettings()" class="primary-button">
                        Save Settings
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- ======================================================== -->
    <!-- 6. FLOATING TOAST CONTAINER                              -->
    <!-- ======================================================== -->
    <div id="toast-container" class="fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none"></div>

    <!-- ======================================================== -->
    <!-- 7. FIXED FEEDBACK TAB (RIGHT EDGE)                       -->
    <!-- ======================================================== -->
    <button
        aria-label="Open feedback"
        onclick="showToast('Thanks for your feedback', 'info')"
        class="fixed bottom-8 right-0 grid h-28 w-11 place-items-center rounded-l-lg bg-[#d83b21] text-white shadow-lg hover:bg-[#bd311a] transition"
    >
        <span class="-rotate-90 text-[11px] font-medium tracking-wide">
            Feedback
        </span>
    </button>

    <!-- Drawer toggle JS helper -->
    <script>
        let isDrawerOpen = true;
        function toggleSidebarDrawer() {
            const drawer = document.getElementById('sidebar-drawer');
            const main = document.getElementById('main-content-area');
            const composer = document.querySelector('.fixed.bottom-6');
            isDrawerOpen = !isDrawerOpen;
            if (isDrawerOpen) {
                drawer.style.transform = 'translateX(0)';
                main.style.left = '372px';
                if (composer) composer.style.left = '396px';
            } else {
                drawer.style.transform = 'translateX(-300px)';
                main.style.left = '72px';
                if (composer) composer.style.left = '96px';
            }
        }
    </script>

    <!-- Main Application Logic (All 63 Functions Intact, Zero Emojis) -->
    <script>
''' + final_script + r'''
    </script>
</body>
</html>
'''

with open('static/index.html', 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"Successfully compiled static/index.html! Length: {len(full_html)} characters.")
