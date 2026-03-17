// /static/app.js
const $ = (sel) => document.querySelector(sel);
const SIDEBAR_COLLAPSED_KEY = "openstoryline_sidebar_collapsed";
const DEVBAR_COLLAPSED_KEY = "openstoryline_devbar_collapsed";
const AUDIO_PREVIEW_MAX = 3;
const CUSTOM_MODEL_KEY = "__custom__";
const SESSION_ID_KEY = "openstoryline_session_id";
const SESSION_LIST_KEY = "openstoryline_session_list_v1";

// =========================================================
// i18n (zh/en) + lang persistence
// =========================================================
const __OS_LANG_STORAGE_KEY = "openstoryline_lang_v1";

const QUICK_PROMPTS = [
  { zh: "详细介绍一下你能做什么", en: "Please describe in detail what you can do." },
  { zh: "帮我找10个夏日海滩素材，剪一个欢快的旅行vlog", en: "Please help me find some summer beach footage and edit it into a 30-second travel vlog." },
  { zh: "我准备长期批量做同类视频，先帮我剪一条示范成片；之后把这套偏好总结成可复用的剪辑风格 Skill。", en: "I plan to produce similar videos in batches over a long period. First, help me edit a sample video; then, help me summarize this set of preferences into a reusable editing style skill."},
  { zh: "根据我的素材内容，仿照鲁迅文风生成文案。", en: "Based on my footage, please generate a Shakespearean-style video script."},
  { zh: "帮我找一些中国春节相关素材，筛选出最有年味的场景，选择喜庆的 BGM", en: "Please help me find some materials related to Chinese New Year, filter out the most festive scenes, and choose celebratory background music."},
];

const __OS_I18N = {
  zh: {
    // topbar
    "main.greeting": "🎬 你好，创作者",
    "topbar.lang_title": "切换语言",
    "topbar.lang_aria": "语言切换",
    "topbar.lang_zh": "中",
    "topbar.lang_en": "EN",
    "topbar.link1": "github 链接",
    "topbar.link2": "使用手册",
    "topbar.node_map": "节点地图",

    // aria
    "aria.sidebar": "侧边栏",
    "aria.sidebar_scroll": "侧边栏滚动区",
    "aria.sidebar_model_select": "对话模型选择",
    "composer.placeholder": "提出任何剪辑需求（Enter 发送，shift + Enter 换行）",
    "assistant.placeholder": "正在调用大模型中…",
    "composer.quick_prompt": "插入提示语",

    // sidebar
    "sidebar.toggle": "收起/展开侧边栏",
    "sidebar.new_chat": "创建新对话",
    "sidebar.history_title": "对话历史",
    "sidebar.history_empty": "暂无历史会话",
    "sidebar.history_aria": "历史会话列表",
    "sidebar.model_label": "对话模型",
    "sidebar.model_select_aria": "选择对话模型",
    "sidebar.custom_model_box_aria": "自定义模型配置",
    "sidebar.custom_model_title": "自定义模型",
    "sidebar.custom_llm_subtitle": "LLM（对话/文案）",
    "sidebar.custom_llm_model_ph": "模型名称，例如 deepseek-chat / gpt-4o-mini",
    "sidebar.custom_llm_baseurl_ph": "Base URL，例如 https://api.xxx.com/v1",
    "sidebar.custom_llm_apikey_ph": "API Key",
    "sidebar.custom_vlm_subtitle": "VLM（素材理解）",
    "sidebar.custom_vlm_model_ph": "模型名称，例如 qwen-vl-plus / gpt-4o",
    "sidebar.custom_vlm_baseurl_ph": "Base URL，例如 https://api.xxx.com/v1",
    "sidebar.custom_vlm_apikey_ph": "API Key",
    "sidebar.custom_hint": "提示：API Key 仅用于本会话的服务端调用；页面与 Tool trace 会自动脱敏，不会显示明文。",
    "sidebar.tts_box_aria": "TTS 服务配置",
    "sidebar.tts_title": "TTS 配置",
    "sidebar.tts_provider_select_aria": "选择 TTS 服务厂家",
    "sidebar.tts_default": "使用默认配置",
    "sidebar.tts_hint": "提示：字段留空将使用 config.toml 中的配置。",
    "sidebar.tts_field_suffix": "（留空则使用服务器默认）",
    "sidebar.use_custom_model": "使用自定义模型",
    "sidebar.llm_label": "LLM 模型",
    "sidebar.vlm_label": "VLM 模型",
    "sidebar.llm_select_aria": "选择 LLM 模型",
    "sidebar.vlm_select_aria": "选择 VLM 模型",
    "sidebar.custom_llm_title": "LLM 自定义模型",
    "sidebar.custom_vlm_title": "VLM 自定义模型",
    "sidebar.custom_llm_box_aria": "LLM 自定义模型配置",
    "sidebar.custom_vlm_box_aria": "VLM 自定义模型配置",

    "sidebar.pexels_box_aria": "Pexels API Key 配置",
    "sidebar.pexels_title": "Pexels 配置",
    "sidebar.pexels_mode_select_aria": "选择 Pexels Key 模式",
    "sidebar.pexels_default": "使用默认配置",
    "sidebar.pexels_custom": "使用自定义 key",
    "sidebar.pexels_apikey_ph": "Pexels API Key",
    "sidebar.pexels_hint": "提示：默认配置会优先使用 config.toml 的 search_media.pexels_api_key；为空时工具内部会从环境变量读取。",

    "sidebar.help.cta": "点击查看配置教程",
    "sidebar.help.llm": "LLM 主要用于对话，在工具内部也被用来生成文案/分组/选择BGM等。",
    "sidebar.help.vlm": "VLM 用于素材理解（图像/视频理解）。自定义时请确认模型支持多模态输入。",
    "sidebar.help.pexels": "Pexels 用于搜索网络素材。免责声明：OpenStoryline 搜索的网络素材均来自Pexels，通过Pexels下载的素材仅用于体验Open-Storyline剪辑效果，不允许再分发或出售。我们只提供工具，所有通过本工具下载和使用的素材（如 Pexels 图像）都由用户自行通过 API 获取，我们不对用户生成的视频内容、素材的合法性或因使用本工具导致的任何版权/肖像权纠纷承担责任。使用时请遵循 Pexels 的许可协议。",
    "sidebar.help.tts": "用于从文案生成配音。",
    "sidebar.help.pexels_home_link": "点击进入 Pexels 官方网站",
    "sidebar.help.pexels_terms_link": "查看 Pexels 用户协议",

    // common
    "common.retry_after_suffix": "（{seconds}s后再试）",

    // toast
    "toast.interrupt_failed": "打断失败：{msg}",
    "toast.pending_limit": "待发送素材已达上限（{max} 个），请先发送/删除后再上传。",
    "toast.pending_limit_partial": "最多还能上传 {remain} 个素材（上限 {max}）。将只上传前 {remain} 个。",
    "toast.uploading": "正在上传素材中… {pct}%",
    "toast.uploading_file": "正在上传素材（{i}/{n}）：{name}… {pct}%",
    "toast.upload_failed": "上传失败：{msg}",
    "toast.delete_failed": "删除失败：{msg}",
    "toast.uploading_cannot_send": "素材正在上传中，上传完成后才能发送。",
    "toast.switch_while_streaming": "正在生成回复，暂时无法切换会话。请先等待完成或打断当前回复。",
    "toast.uploading_interrupt_send": "素材正在上传中，暂时无法发送新消息。已为你打断当前回复；上传完成后再按 Enter 发送。",

    // tools
    "tool.card.default_name": "工具调用",
    "tool.card.fallback_name": "MCP 工具",

    "tool.preview.render_title": "成片预览",
    "tool.preview.other_videos": "其它视频（点击预览）",
    "tool.preview.videos": "视频（点击预览）",
    "tool.preview.images": "图片（点击预览）",
    "tool.preview.audio": "音频",
    "tool.preview.listen": "试听",
    "tool.preview.split_shots": "镜头切分结果（点击预览）",

    "tool.preview.btn_modal": "弹窗预览",
    "tool.preview.btn_open": "打开",

    "tool.preview.more_items": "还有 {n} 个未展示",
    "tool.preview.more_audios": "还有 {n} 个音频未展示",

    "tool.preview.label.audio": "音频 {i}",
    "tool.preview.label.video": "视频 {i}",
    "tool.preview.label.image": "图片 {i}",
    "tool.preview.label.shot": "镜头 {i}",

    "preview.unsupported": "该类型暂不支持内嵌预览：",
    "preview.open_download": "打开/下载",
  },
  en: {
    // topbar
    "main.greeting": "🎬 Hi, creator",
    "topbar.lang_title": "Switch language",
    "topbar.lang_aria": "Language switch",
    "topbar.lang_zh": "中",
    "topbar.lang_en": "EN",
    "topbar.link1": "github link",
    "topbar.link2": "user guide",
    "topbar.node_map": "node map",

    // aria
    "aria.sidebar": "Sidebar",
    "aria.sidebar_scroll": "Sidebar scroll area",
    "aria.sidebar_model_select": "Chat model selector",
    "composer.placeholder": "Make any editing requests (Enter to send, Shift + Enter for line break)",
    "assistant.placeholder": "Calling the LLM…",
    "composer.quick_prompt": "Insert a preset prompt",

    // sidebar
    "sidebar.toggle": "Collapse/expand sidebar",
    "sidebar.new_chat": "New chat",
    "sidebar.history_title": "History",
    "sidebar.history_empty": "No past chats yet",
    "sidebar.history_aria": "Chat history list",
    "sidebar.model_label": "Chat model",
    "sidebar.model_select_aria": "Select chat model",
    "sidebar.custom_model_box_aria": "Custom model settings",
    "sidebar.custom_model_title": "Custom model",
    "sidebar.custom_llm_subtitle": "LLM (chat/copywriting)",
    "sidebar.custom_llm_model_ph": "Model name, e.g. deepseek-chat / gpt-4o-mini",
    "sidebar.custom_llm_baseurl_ph": "Base URL, e.g. https://api.xxx.com/v1",
    "sidebar.custom_llm_apikey_ph": "API key",
    "sidebar.custom_vlm_subtitle": "VLM (media understanding)",
    "sidebar.custom_vlm_model_ph": "Model name, e.g. qwen-vl-plus / gpt-4o",
    "sidebar.custom_vlm_baseurl_ph": "Base URL, e.g. https://api.xxx.com/v1",
    "sidebar.custom_vlm_apikey_ph": "API key",
    "sidebar.custom_hint": "Note: API keys are used only for server-side calls in this session. They are masked in the UI and tool trace.",
    "sidebar.tts_box_aria": "TTS configuration",
    "sidebar.tts_title": "TTS",
    "sidebar.tts_provider_select_aria": "Select a TTS provider",
    "sidebar.tts_default": "Use default configuration",
    "sidebar.tts_hint": "Note: leaving fields empty will fall back to config.toml.",
    "sidebar.tts_field_suffix": " (leave empty to use server default)",
    "sidebar.use_custom_model": "Use custom model",
    "sidebar.llm_label": "LLM model",
    "sidebar.vlm_label": "VLM model",
    "sidebar.llm_select_aria": "Select LLM model",
    "sidebar.vlm_select_aria": "Select VLM model",
    "sidebar.custom_llm_title": "Custom LLM",
    "sidebar.custom_vlm_title": "Custom VLM",
    "sidebar.custom_llm_box_aria": "Custom LLM settings",
    "sidebar.custom_vlm_box_aria": "Custom VLM settings",

    "sidebar.pexels_box_aria": "Pexels API key settings",
    "sidebar.pexels_title": "Pexels",
    "sidebar.pexels_mode_select_aria": "Select Pexels key mode",
    "sidebar.pexels_default": "Use default configuration",
    "sidebar.pexels_custom": "Use custom key",
    "sidebar.pexels_apikey_ph": "Pexels API key",
    "sidebar.pexels_hint": "Note: default mode prefers config.toml (search_media.pexel_api_key). If empty, the tool will fall back to environment variables.",

    "sidebar.help.cta": "Click to view the configuration guide",
    "sidebar.help.llm": "LLM is used for chat/copywriting.",
    "sidebar.help.vlm": "VLM is used for media understanding (image/video).",
    "sidebar.help.pexels": "Pexels is used for media search. Disclaimer: The online content searched by OpenStoryline is all from Pexels. Footage downloaded via Pexels is for the sole purpose of experiencing Open-Storyline editing effects and may not be redistributed or sold. We only provide the tool. All materials downloaded and used through this tool (such as Pexels images) are obtained by the user through the API. We are not responsible for the legality of user-generated video content or materials, or for any copyright/portrait rights disputes arising from the use of this tool. Please comply with the Pexels license agreement when using it.",
    "sidebar.help.tts": "TTS is used to generate voiceover from text.",
    "sidebar.help.pexels_home_link": "Visit the official Pexels website",
    "sidebar.help.pexels_terms_link": "View Pexels Terms",

    // common
    "common.retry_after_suffix": " (retry in {seconds}s)",

    // toast
    "toast.interrupt_failed": "Interrupt failed: {msg}",
    "toast.pending_limit": "Pending media limit reached ({max}). Please send/delete before uploading more.",
    "toast.pending_limit_partial": "You can upload at most {remain} more file(s) (limit {max}). Only the first {remain} will be uploaded.",
    "toast.uploading": "Uploading media… {pct}%",
    "toast.uploading_file": "Uploading ({i}/{n}): {name}… {pct}%",
    "toast.upload_failed": "Upload failed: {msg}",
    "toast.delete_failed": "Delete failed: {msg}",
    "toast.uploading_cannot_send": "Media is uploading. Please wait until it finishes before sending.",
    "toast.switch_while_streaming": "A reply is still being generated. Please wait or interrupt before switching chats.",
    "toast.uploading_interrupt_send": "Media is uploading, so a new message can't be sent yet. I interrupted the current reply; press Enter after the upload finishes.",
    
    // tools
    "tool.card.default_name": "Tool call",
    "tool.card.fallback_name": "MCP Tool",

    "tool.preview.render_title": "Rendered preview",
    "tool.preview.other_videos": "Other videos (click to preview)",
    "tool.preview.videos": "Videos (click to preview)",
    "tool.preview.images": "Images (click to preview)",
    "tool.preview.audio": "Audio",
    "tool.preview.listen": "Listen",
    "tool.preview.split_shots": "Shot splitting results (click to preview)",

    "tool.preview.btn_modal": "Open preview",
    "tool.preview.btn_open": "Open",

    "tool.preview.more_items": "{n} more not shown",
    "tool.preview.more_audios": "{n} more audio clip(s) not shown",

    "tool.preview.label.audio": "Audio {i}",
    "tool.preview.label.video": "Video {i}",
    "tool.preview.label.image": "Image {i}",
    "tool.preview.label.shot": "Shot {i}",

    "preview.unsupported": "This type can't be previewed inline:",
    "preview.open_download": "Open/Download",
  }
};

function __osNormLang(x) {
  const s = String(x || "").trim().toLowerCase();
  if (s === "en" || s.startsWith("en-")) return "en";
  return "zh";
}

function __osLoadLang() {
  try {
    const v = localStorage.getItem(__OS_LANG_STORAGE_KEY);
    return v ? __osNormLang(v) : null;
  } catch {
    return null;
  }
}

function __osSaveLang(lang) {
  try { localStorage.setItem(__OS_LANG_STORAGE_KEY, lang); } catch {}
}

function __osFormat(tpl, vars) {
  const s = String(tpl ?? "");
  return s.replace(/\{(\w+)\}/g, (_, k) => {
    if (!vars || vars[k] == null) return "";
    return String(vars[k]);
  });
}

function __t(key, vars) {
  const lang = __osNormLang(window.OPENSTORYLINE_LANG || "zh");
  const table = __OS_I18N[lang] || __OS_I18N.zh;
  const raw = (table && table[key] != null) ? table[key] : (__OS_I18N.zh[key] ?? key);
  return __osFormat(raw, vars);
}

function __applyI18n(root = document) {
  // textContent
  root.querySelectorAll("[data-i18n]").forEach((el) => {
    const k = el.getAttribute("data-i18n");
    if (!k) return;
    el.textContent = __t(k);
  });

  // attributes
  root.querySelectorAll("[data-i18n-title]").forEach((el) => {
    const k = el.getAttribute("data-i18n-title");
    if (!k) return;
    el.setAttribute("title", __t(k));
  });

  root.querySelectorAll("[data-i18n-aria-label]").forEach((el) => {
    const k = el.getAttribute("data-i18n-aria-label");
    if (!k) return;
    el.setAttribute("aria-label", __t(k));
  });

  root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const k = el.getAttribute("data-i18n-placeholder");
    if (!k) return;
    el.setAttribute("placeholder", __t(k));
  });
}

// TTS 动态字段 placeholder（suffix）重渲染：
// - 创建 input 时会写入 data-os-ph-base / data-os-ph-suffix
function __rerenderTtsFieldPlaceholders(root = document) {
  root.querySelectorAll("input[data-os-ph-base]").forEach((el) => {
    const base = String(el.getAttribute("data-os-ph-base") || "");
    const needSuffix = el.getAttribute("data-os-ph-suffix") === "1";
    el.setAttribute("placeholder", needSuffix ? `${base}${__t("sidebar.tts_field_suffix")}` : base);
  });
}

function __osApplyHelpLinks(root = document) {
  const lang = __osNormLang(window.OPENSTORYLINE_LANG || "zh");
  const nodes = (root || document).querySelectorAll(".sidebar-help[data-help-zh], .sidebar-help[data-help-en]");

  nodes.forEach((a) => {
    const zh = a.getAttribute("data-help-zh") || "";
    const en = a.getAttribute("data-help-en") || "";
    const href = (lang === "en") ? (en || zh) : (zh || en);
    if (href) a.setAttribute("href", href);
  });
}

function __osApplyTooltipLinks(root = document) {
  const lang = __osNormLang(window.OPENSTORYLINE_LANG || "zh");

  const nodes = (root || document).querySelectorAll(
    ".sidebar-help-tooltip-link[data-terms-zh], .sidebar-help-tooltip-link[data-terms-en], " +
    ".sidebar-help-tooltip-link[data-pexels-home-zh], .sidebar-help-tooltip-link[data-pexels-home-en]"
  );

  const pickHref = (el) => {
    const homeZh = el.getAttribute("data-pexels-home-zh") || "";
    const homeEn = el.getAttribute("data-pexels-home-en") || "";
    const termsZh = el.getAttribute("data-terms-zh") || "";
    const termsEn = el.getAttribute("data-terms-en") || "";

    const zh = homeZh || termsZh;
    const en = homeEn || termsEn;

    return (lang === "en") ? (en || zh) : (zh || en);
  };

  const open = (el, ev) => {
    if (ev) {
      ev.preventDefault();
      ev.stopPropagation();
    }
    const href = pickHref(el);
    if (!href) return;
    window.open(href, "_blank", "noopener,noreferrer");
  };

  nodes.forEach((el) => {
    if (el.__osTooltipLinkBound) return;
    el.__osTooltipLinkBound = true;

    el.addEventListener("click", (e) => open(el, e), true);

    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") open(el, e);
    }, true);
  });
}

function __osEnsureLeadingSlash(s) {
  s = String(s ?? "").trim();
  if (!s) return "";
  return s.startsWith("/") ? s : ("/" + s);
}


function __osAppendToCurrentUrl(suffix) {
  const suf = __osEnsureLeadingSlash(suffix);
  if (!suf) return "";

  const u = new URL(window.location.href);

  const h = String(u.hash || "");
  if (h.startsWith("#/") || h.startsWith("#!/")) {
    const isBang = h.startsWith("#!/");
    const route = isBang ? h.slice(2) : h.slice(1); // "/xxx..."
    const routeNoTrail = route.replace(/\/+$/, "");

    if (routeNoTrail.endsWith(suf)) return `${u.origin}${u.pathname}${isBang ? "#!" : "#"}${routeNoTrail}`;

    return `${u.origin}${u.pathname}${isBang ? "#!" : "#"}${routeNoTrail}${suf}`;
  }
  u.search = "";
  u.hash = "";

  let path = u.pathname || "/";

  if (!path.endsWith("/")) {
    const last = path.split("/").pop() || "";
    if (last.includes(".")) {
      path = path.slice(0, path.length - last.length); // 留下末尾的 "/"
    }
  }

  const base = `${u.origin}${path}`.replace(/\/+$/, "");
  return `${base}${suf}`;
}

function __osApplyTopbarLinks(root = document) {
  const lang = __osNormLang(window.OPENSTORYLINE_LANG || "zh");
  const nodes = (root || document).querySelectorAll(
    ".topbar-link[data-link-zh], .topbar-link[data-link-en], .topbar-link[data-link-suffix], .topbar-link[data-link-suffix-zh], .topbar-link[data-link-suffix-en]"
  );

  nodes.forEach((a) => {
    // 1) 动态 suffix：优先
    const sufZh = a.getAttribute("data-link-suffix-zh") || "";
    const sufEn = a.getAttribute("data-link-suffix-en") || "";
    const suf = a.getAttribute("data-link-suffix") || "";

    const pickedSuffix = (lang === "en") ? (sufEn || sufZh || suf) : (sufZh || sufEn || suf);
    if (pickedSuffix) {
      const href = __osAppendToCurrentUrl(pickedSuffix);
      if (href) a.setAttribute("href", href);
      return;
    }

    // 2) 静态 zh/en URL
    const zh = a.getAttribute("data-link-zh") || "";
    const en = a.getAttribute("data-link-en") || "";
    const href = (lang === "en") ? (en || zh) : (zh || en);
    if (href) a.setAttribute("href", href);
  });
}


function __applyLang(lang, { persist = true } = {}) {
  const v = __osNormLang(lang);
  window.OPENSTORYLINE_LANG = v;

  if (persist) __osSaveLang(v);

  document.body.classList.toggle("lang-en", v === "en");
  document.body.classList.toggle("lang-zh", v === "zh");
  document.documentElement.lang = (v === "en") ? "en" : "zh-CN";

  __applyI18n(document);
  __rerenderTtsFieldPlaceholders(document);
  __osApplyHelpLinks(document);
  __osApplyTopbarLinks(document);
  __osApplyTooltipLinks(document);
}

// init once
(() => {
  const stored = __osLoadLang();
  const initial = stored || __osNormLang(document.documentElement.lang || "zh");
  __applyLang(initial, { persist: stored != null }); // 有存储就保留；没存储就不写入
})();


class ApiClient {
  async createSession() {
    const r = await fetch("/api/sessions", { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  async getSession(sessionId) {
    const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
    if (!r.ok) {
      const err = new Error(await this._readFetchError(r));
      err.status = r.status;
      throw err;
    }
    return await r.json();
  }

  async getTtsUiSchema() {
    const r = await fetch("/api/meta/tts", { method: "GET" });
    if (!r.ok) throw new Error(await this._readFetchError(r));
    return await r.json(); // { default_provider, providers:[...] }
  }

  async cancelTurn(sessionId) {
    const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/cancel`, { method: "POST" });
    if (!r.ok) throw new Error(await this._readFetchError(r));
    return await r.json();
  }

  async _readFetchError(r) {
    const t = await r.text();
    try {
      const j = JSON.parse(t);
      // 兼容 middleware/接口的 429: {detail:"Too Many Requests", retry_after:n}
      if (j && typeof j === "object") {
        const ra = (j.retry_after != null) ? Number(j.retry_after) : (j.detail && j.detail.retry_after != null ? Number(j.detail.retry_after) : null);

        if (typeof j.detail === "string") return ra != null ? `${j.detail}${__t("common.retry_after_suffix", { seconds: ra })}` : j.detail;
        if (j.detail && typeof j.detail === "object") {
          const msg = j.detail.message || j.detail.detail || j.detail.error || JSON.stringify(j.detail);
          return ra != null ? `${msg}${__t("common.retry_after_suffix", { seconds: ra })}` : msg;
        }
        if (typeof j.message === "string") return ra != null ? `${j.message}${__t("common.retry_after_suffix", { seconds: ra })}` : j.message;
      }
    } catch {}
    return t || `HTTP ${r.status}`;
  }

  async initResumableMedia(sessionId, file, { chunkSize } = {}) {
    const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/media/init`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        size: file.size,
        mime_type: file.type,
        last_modified: file.lastModified,
        chunk_size: chunkSize, // 服务端可忽略（以服务端配置为准）
      }),
    });
    if (!r.ok) throw new Error(await this._readFetchError(r));
    return await r.json();
  }

  uploadResumableChunk(sessionId, uploadId, index, blob, onProgress) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("index", String(index));
      // 这里用 blob（分片），而不是整文件
      form.append("chunk", blob, "chunk");

      const xhr = new XMLHttpRequest();
      xhr.open(
        "POST",
        `/api/sessions/${encodeURIComponent(sessionId)}/media/${encodeURIComponent(uploadId)}/chunk`,
        true
      );

      xhr.upload.onprogress = (e) => {
        if (typeof onProgress === "function") {
          const loaded = e && typeof e.loaded === "number" ? e.loaded : 0;
          const total = e && typeof e.total === "number" ? e.total : (blob ? blob.size : 0);
          onProgress(loaded, total);
        }
      };

      xhr.onload = () => {
        const ok = xhr.status >= 200 && xhr.status < 300;
        if (ok) {
          try { resolve(JSON.parse(xhr.responseText || "{}")); }
          catch (e) { resolve({}); }
          return;
        }

        // 错误：尽量把 JSON detail 解析成可读信息
        const text = xhr.responseText || "";
        let msg = text || `HTTP ${xhr.status}`;
        try {
          const j = JSON.parse(text);
          const ra = (j && typeof j === "object" && j.retry_after != null) ? Number(j.retry_after) : null;
          if (j && typeof j.detail === "string") msg = ra != null ? `${j.detail}${__t("common.retry_after_suffix", { seconds: ra })}` : j.detail;
          else if (j && typeof j.detail === "object") {
            const m = j.detail.message || j.detail.detail || j.detail.error || JSON.stringify(j.detail);
            msg = ra != null ? `${m}${__t("common.retry_after_suffix", { seconds: ra })}` : m;
          }
        } catch {}
        reject(new Error(msg));
      };

      xhr.onerror = () => reject(new Error("network error"));
      xhr.send(form);
    });
  }

  async completeResumableMedia(sessionId, uploadId) {
    const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/media/${encodeURIComponent(uploadId)}/complete`, {
      method: "POST",
    });
    if (!r.ok) throw new Error(await this._readFetchError(r));
    return await r.json(); // { media, pending_media }
  }

  async cancelResumableMedia(sessionId, uploadId) {
    try {
      await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/media/${encodeURIComponent(uploadId)}/cancel`, { method: "POST" });
    } catch {}
  }

  // 单文件：init -> chunk... -> complete
  async uploadMediaChunked(sessionId, file, { chunkSize, onProgress } = {}) {
    const init = await this.initResumableMedia(sessionId, file, { chunkSize });
    const uploadId = init.upload_id;
    const cs = Number(init.chunk_size) || Number(chunkSize) || (32 * 1024 * 1024);

    const totalChunks = Number(init.total_chunks) || Math.ceil((file.size || 0) / cs) || 1;

    let confirmed = 0; // 已完成分片字节数（本文件内）
    try {
      for (let i = 0; i < totalChunks; i++) {
        const start = i * cs;
        const end = Math.min(file.size, start + cs);
        const blob = file.slice(start, end);

        await this.uploadResumableChunk(sessionId, uploadId, i, blob, (loaded) => {
          if (typeof onProgress === "function") {
            // confirmed + 当前分片已上传字节
            onProgress(Math.min(file.size, confirmed + (loaded || 0)), file.size);
          }
        });

        confirmed += blob.size;
        if (typeof onProgress === "function") onProgress(Math.min(file.size, confirmed), file.size);
      }

      return await this.completeResumableMedia(sessionId, uploadId);
    } catch (e) {
      // 失败尽量清理服务端临时文件
      await this.cancelResumableMedia(sessionId, uploadId);
      throw e;
    }
  }


  async deletePendingMedia(sessionId, mediaId) {
    const r = await fetch(
      `/api/sessions/${encodeURIComponent(sessionId)}/media/pending/${encodeURIComponent(mediaId)}`,
      { method: "DELETE" }
    );
    if (!r.ok) throw new Error(await r.text());
    return await r.json();
  }
}

class WsClient {
  constructor(url, onEvent) {
    this.url = url;
    this.onEvent = onEvent;
    this.ws = null;
    this._timer = null;
    this._closedByUser = false;
  }

  connect() {
    this._closedByUser = false;
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      // 心跳（可选）
      this._timer = setInterval(() => {
        if (this.ws && this.ws.readyState === 1) {
          this.send("ping", {});
        }
      }, 25000);
    };

    this.ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (this.onEvent) this.onEvent(msg);
    };

    this.ws.onclose = (ev) => {
      if (this._timer) clearInterval(this._timer);
      this._timer = null;

      console.warn("[ws] closed", {
        code: ev?.code,
        reason: ev?.reason,
        wasClean: ev?.wasClean,
      });
      
      if (this._closedByUser) return;

      // session 不存在就不要重连
      if (ev && ev.code === 4404) {
        localStorage.removeItem(SESSION_ID_KEY);
        location.reload();
        return;
      }

      setTimeout(() => this.connect(), 1000);
    };
  }

  close() {
    this._closedByUser = true;
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
    if (this.ws) {
      try { this.ws.close(1000, "client switch session"); } catch {}
      this.ws = null;
    }
  }

  send(type, data) {
    if (!this.ws || this.ws.readyState !== 1) return;
    this.ws.send(JSON.stringify({ type, data }));
  }
}

class ChatUI {
  constructor() {
    this.chatEl = $("#chat");
    this.pendingBarEl = $("#pendingBar");
    this.pendingRowEl = $("#pendingRow");
    this.toastEl = $("#toast");
    // developer
    this.devLogEl = $("#devLog")
    this.devDomByID = new Map()

    this.modalEl = $("#modal");
    this.modalBackdrop = $("#modalBackdrop");
    this.modalClose = $("#modalClose");
    this.modalContent = $("#modalContent");

    this.toolDomById = new Map();
    this.toolMediaDomById = new Map();
    this.currentAssistant = null; // { bubbleEl, rawText }

    this.mdStreaming = true;          // 是否启用流式 markdown
    this._mdRaf = 0;                  // requestAnimationFrame id
    this._mdTimer = null;             // setTimeout id
    this._mdLastRenderAt = 0;         // 上次渲染时间
    this._mdRenderInterval = 80;      // 渲染时间间隔

    this._toolUi = this._loadToolUiConfig();

    this.scrollBtnEl = $("#scrollToBottomBtn");
    this._bindScrollJumpBtn();
    this._bindScrollWatcher();

    this._toastI18n = null;
  }

  setSessionId(sessionId) {
    this._sessionId = sessionId;
    const s = `session_id: ${sessionId}`;
    const el = $("#sidebarSid");
    if (el) el.textContent = s;
  }

  _setToastText(text) {
    this.toastEl.textContent = String(text ?? "");
    this.toastEl.classList.remove("hidden");
  }

  showToast(text) {
    this._toastI18n = null;
    this._setToastText(text);
  }

  showToastI18n(key, vars) {
    this._toastI18n = { key: String(key || ""), vars: vars || {} };
    this._setToastText(__t(key, vars));
  }

  rerenderToast() {
    if (!this.toastEl || this.toastEl.classList.contains("hidden")) return;
    if (!this._toastI18n || !this._toastI18n.key) return;
    this._setToastText(__t(this._toastI18n.key, this._toastI18n.vars));
  }

  rerenderAssistantPlaceholder() {
    const cur = this.currentAssistant;
    if (!cur || !cur.bubbleEl) return;

    if ((cur.rawText || "").trim()) return;

    const key = cur._placeholderKey;
    if (!key) return;

    this.setBubbleContent(cur.bubbleEl, __t(key));
  }


  hideToast() {
    this.toastEl.classList.add("hidden");
  }


  _docScrollHeight() {
    const de = document.documentElement;
    return (de && de.scrollHeight) ? de.scrollHeight : document.body.scrollHeight;
  }

  isNearBottom(threshold = 160) {
    const top = window.scrollY || window.pageYOffset || 0;
    const h = window.innerHeight || 0;
    return (top + h) >= (this._docScrollHeight() - threshold);
  }

  _updateScrollJumpBtnVisibility(force) {
    if (!this.scrollBtnEl) return;

    let show;
    if (force === true) show = true;
    else if (force === false) show = false;
    else show = !this.isNearBottom();

    this.scrollBtnEl.classList.toggle("hidden", !show);
  }

  scrollToBottom({ behavior = "smooth" } = {}) {
    requestAnimationFrame(() => {
      window.scrollTo({ top: this._docScrollHeight(), behavior });
    });
  }

  maybeAutoScroll(wasNearBottom, { behavior = "auto" } = {}) {
    if (wasNearBottom) {
      this.scrollToBottom({ behavior });
      this._updateScrollJumpBtnVisibility(false);
    } else {
      this._updateScrollJumpBtnVisibility(true);
    }
  }

  _bindScrollJumpBtn() {
    if (!this.scrollBtnEl || this._scrollBtnBound) return;
    this._scrollBtnBound = true;

    this.scrollBtnEl.addEventListener("click", (e) => {
      e.preventDefault();
      this.scrollToBottom({ behavior: "smooth" });
      this._updateScrollJumpBtnVisibility(false);
    });
  }

  _bindScrollWatcher() {
    if (this._scrollWatchBound) return;
    this._scrollWatchBound = true;

    const handler = () => this._updateScrollJumpBtnVisibility();
    window.addEventListener("scroll", handler, { passive: true });
    window.addEventListener("resize", handler, { passive: true });

    requestAnimationFrame(handler);
  }


  clearAll() {
    this.chatEl.innerHTML = "";

    // 停掉所有假进度条 timer
    for (const [, dom] of this.toolDomById) {
      if (dom && dom._fakeTimer) {
        clearInterval(dom._fakeTimer);
        dom._fakeTimer = null;
      }
    }

    this.toolDomById.clear();
    this.currentAssistant = null;

    if (this.devLogEl) this.devLogEl.innerHTML = "";
    this.devDomByID.clear()

    // 清掉 tool 外部媒体块
    if (this.toolMediaDomById) {
      for (const [, dom] of this.toolMediaDomById) {
        try { dom?.wrap?.remove(); } catch {}
      }
      this.toolMediaDomById.clear();
    }

  }
  
  setBubbleContent(bubbleEl, text, { markdown = true } = {}) {
    const s = String(text ?? "");

    // 纯文本模式：用于 user bubble（避免 marked 生成 <p> 导致默认 margin 撑大气泡）
    if (!markdown || !window.marked || !window.DOMPurify) {
      bubbleEl.textContent = s;
      return;
    }

    if (!this._mdInited) {
      window.marked.setOptions({
        gfm: true,
        breaks: true,
        headerIds: false,
        mangle: false,
      });

      window.DOMPurify.addHook("afterSanitizeAttributes", (node) => {
        if (node.tagName === "A") {
          node.setAttribute("target", "_blank");
          node.setAttribute("rel", "noopener noreferrer");
        }
      });

      this._mdInited = true;
    }

    const rawHtml = window.marked.parse(s);
    const safeHtml = window.DOMPurify.sanitize(rawHtml, { USE_PROFILES: { html: true } });
    bubbleEl.innerHTML = safeHtml;
  }


  renderPendingMedia(pendingMedia) {
    this.pendingRowEl.innerHTML = "";
    if (!pendingMedia || !pendingMedia.length) {
      this.pendingBarEl.classList.add("hidden");
      return;
    }
    this.pendingBarEl.classList.remove("hidden");

    for (const a of pendingMedia) {
      this.pendingRowEl.appendChild(this.renderMediaThumb(a, { removable: true }));
    }
  }

  mediaTag(kind) {
    if (kind === "image") return "IMG";
    if (kind === "video") return "VID";
    return "";
  }

  renderMediaThumb(media, { removable } = { removable: false }) {
    const el = document.createElement("div");
    el.className = "media-item";
    el.title = media.name || "";

    const img = document.createElement("img");
    img.src = media.thumb_url;
    img.alt = media.name || "";
    el.appendChild(img);

    const tag = document.createElement("div");
    tag.className = "media-tag";
    tag.textContent = this.mediaTag(media.kind);
    el.appendChild(tag);

    if (media.kind === "video") {
      const play = document.createElement("div");
      play.className = "media-play";
      el.appendChild(play);
    }

    el.addEventListener("click", (e) => {
      if (e.target?.classList?.contains("media-remove")) return;
      this.openPreview(media);
    });

    if (removable) {
      const rm = document.createElement("div");
      rm.className = "media-remove";
      rm.textContent = "×";
      rm.dataset.mediaId = media.id;
      el.appendChild(rm);
    }

    return el;
  }

  renderAttachmentsRow(attachments, alignRight) {
    if (!attachments || !attachments.length) return null;

    const wrap = document.createElement("div");
    wrap.className = "attach-wrap";
    if (alignRight) wrap.classList.add("align-right");

    const row = document.createElement("div");
    row.className = "attach-row";

    for (const a of attachments) {
      row.appendChild(this.renderMediaThumb(a, { removable: false }));
    }

    wrap.appendChild(row);
    return wrap;
  }

  appendUserMessage(text, attachments) {
    const wrap = document.createElement("div");
    wrap.className = "msg user";

    const container = document.createElement("div");
    container.style.maxWidth = "78%";

    const attachRow = this.renderAttachmentsRow(attachments, true);
    if (attachRow) container.appendChild(attachRow);

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    this.setBubbleContent(bubble, text, { markdown: false });
    container.appendChild(bubble);

    wrap.appendChild(container);
    this.chatEl.appendChild(wrap);
    this.scrollToBottom({ behavior: "smooth" });
    this._updateScrollJumpBtnVisibility(false);
  }

  startAssistantMessage({ placeholder = true } = {}) {
    const wasNearBottom = this.isNearBottom();
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const phKey = "assistant.placeholder";
    if (placeholder) {
      this.setBubbleContent(bubble, __t(phKey));
    } else {
      this.setBubbleContent(bubble, "");
    }

    wrap.appendChild(bubble);
    this.chatEl.appendChild(wrap);
    this.maybeAutoScroll(wasNearBottom, { behavior: "auto" });

    this.currentAssistant = {
      wrapEl: wrap,
      bubbleEl: bubble,
      rawText: "",
      _placeholderKey: placeholder ? phKey : null,
    };
  }




  _normalizeStreamingMarkdown(s) {
    s = String(s ?? "").replace(/\r\n?/g, "\n");

    const ticks = (s.match(/```/g) || []).length;
    if (ticks % 2 === 1) s += "\n```";

    return s;
  }

  _renderAssistantStreaming(cur) {
    this._mdLastRenderAt = Date.now();

    const wasNearBottom = this.isNearBottom(160);

    const md = this._normalizeStreamingMarkdown(cur.rawText);
    this.setBubbleContent(cur.bubbleEl, md);

    if (wasNearBottom) this.scrollToBottom({ behavior: "auto" });
    else this._updateScrollJumpBtnVisibility(true); 
  }

  appendAssistantDelta(delta) {
    console.log("md deps", !!window.marked, !!window.DOMPurify);

    if (!this.currentAssistant) this.startAssistantMessage({ placeholder: false });

    const cur = this.currentAssistant;
    cur.rawText += (delta || "");

    // 节流：避免每 token 都 parse + sanitize
    const now = Date.now();
    const due = now - this._mdLastRenderAt >= this._mdRenderInterval;

    if (due) {
      this._renderAssistantStreaming(cur);
      return;
    }

    if (this._mdTimer) return;
    const wait = Math.max(0, this._mdRenderInterval - (now - this._mdLastRenderAt));
    this._mdTimer = setTimeout(() => {
      this._mdTimer = null;
      if (this.currentAssistant) this._renderAssistantStreaming(this.currentAssistant);
    }, wait);
  }

  finalizeAssistant(text) {
    const wasNearBottom = this.isNearBottom();
    if (!this.currentAssistant) {
      this.startAssistantMessage({ placeholder: false});
    }
    const cur = this.currentAssistant;
    cur.rawText = (text ?? cur.rawText ?? "").trim();
    this.setBubbleContent(cur.bubbleEl, cur.rawText || "（未生成最终答复）");
    this.currentAssistant = null;
    this.maybeAutoScroll(wasNearBottom, { behavior: "auto" });
  }

  // 结束当前 assistant 分段（用于 tool.start 前封口）
  flushAssistantSegment() {
    const wasNearBottom = this.isNearBottom();
    const cur = this.currentAssistant;
    if (!cur) return;

    const text = (cur.rawText || "").trim();
    if (!text) {
      // 没有任何 token（只有占位文案）=> 直接移除
      if (cur.wrapEl) cur.wrapEl.remove();
    } else {
      this.setBubbleContent(cur.bubbleEl, text);
    }

    this.currentAssistant = null;
    this.maybeAutoScroll(wasNearBottom, { behavior: "auto" });
  }

  // 结束整个 turn（对应后端 assistant.end）
  endAssistantTurn(text) {
    const wasNearBottom = this.isNearBottom();
    const s = String(text ?? "").trim();

    if (this.currentAssistant) {
      const cur = this.currentAssistant;

      // 如果服务端给了最终文本，以服务端为准
      if (s) cur.rawText = s;

      const finalText = (cur.rawText || "").trim();
      if (!finalText) {
        if (cur.wrapEl) cur.wrapEl.remove();
      } else {
        this.setBubbleContent(cur.bubbleEl, finalText);
      }

      this.currentAssistant = null;
      this.maybeAutoScroll(wasNearBottom, { behavior: "auto" });
      return;
    }

    // 没有正在流的 bubble：只有当确实有文本时才新建一条
    if (s) {
      this.startAssistantMessage({ placeholder: false });
      const cur = this.currentAssistant;
      cur.rawText = s;
      this.setBubbleContent(cur.bubbleEl, s);
      this.currentAssistant = null;
      this.scrollToBottom();
    }
  }

  _loadToolUiConfig() {
    const cfg = (window.OPENSTORYLINE_TOOL_UI && typeof window.OPENSTORYLINE_TOOL_UI === "object")
      ? window.OPENSTORYLINE_TOOL_UI
      : {};

    const labels =
      (cfg.labels && typeof cfg.labels === "object") ? cfg.labels :
      (window.OPENSTORYLINE_TOOL_LABELS && typeof window.OPENSTORYLINE_TOOL_LABELS === "object") ? window.OPENSTORYLINE_TOOL_LABELS :
      {};

    const estimatesMs =
      (cfg.estimates_ms && typeof cfg.estimates_ms === "object") ? cfg.estimates_ms :
      (cfg.estimatesMs && typeof cfg.estimatesMs === "object") ? cfg.estimatesMs :
      (window.OPENSTORYLINE_TOOL_ESTIMATES && typeof window.OPENSTORYLINE_TOOL_ESTIMATES === "object") ? window.OPENSTORYLINE_TOOL_ESTIMATES :
      {};

    const defaultEstimateMs = Number(cfg.default_estimate_ms ?? cfg.defaultEstimateMs ?? 8000);
    const tickMs = Number(cfg.tick_ms ?? cfg.tickMs ?? 120);
    const capRunning = Number(cfg.cap_running_progress ?? cfg.capRunningProgress ?? 0.99);

    return {
      labels,
      estimatesMs,
      defaultEstimateMs: (Number.isFinite(defaultEstimateMs) && defaultEstimateMs > 0) ? defaultEstimateMs : 8000,
      tickMs: (Number.isFinite(tickMs) && tickMs >= 30) ? tickMs : 120,
      capRunningProgress: (Number.isFinite(capRunning) && capRunning > 0 && capRunning < 1) ? capRunning : 0.99,

      // autoOpenWhileRunning: (cfg.auto_open_while_running != null) ? !!cfg.auto_open_while_running : false,
      // autoCollapseOnDone: (cfg.auto_collapse_on_done != null) ? !!cfg.auto_collapse_on_done : false,

      hideRawToolName: (cfg.hide_raw_tool_name != null) ? !!cfg.hide_raw_tool_name : true,
      showRawToolNameInDev: (cfg.show_raw_tool_name_in_dev != null) ? !!cfg.show_raw_tool_name_in_dev : false,
    };
  }

  _toolFullName(server, name) {
    return `${server || ""}.${name || ""}`.replace(/^\./, "");
  }

  _toolDisplayName(server, name) {
    const full = this._toolFullName(server, name);
    const labels = (this._toolUi && this._toolUi.labels) || {};

    const hit =
      labels[full] ??
      labels[name] ??
      labels[String(full).toLowerCase()] ??
      labels[String(name).toLowerCase()];

    if (hit != null) {
      if (typeof hit === "string") return String(hit);

      if (hit && typeof hit === "object") {
        const lang = __osNormLang(window.OPENSTORYLINE_LANG || "zh");
        const v = hit[lang] ?? hit.zh ?? hit.en;
        if (v != null) return String(v);
      }
    }

    if (this._toolUi && this._toolUi.hideRawToolName) return __t("tool.card.default_name");
    return full || __t("tool.card.fallback_name");
  }

  _toolEstimateMs(server, name) {
    const full = this._toolFullName(server, name);
    const map = (this._toolUi && this._toolUi.estimatesMs) || {};
    const v = map[full] ?? map[name];
    const ms = Number(v);
    if (Number.isFinite(ms) && ms > 0) return ms;
    return (this._toolUi && this._toolUi.defaultEstimateMs) ? this._toolUi.defaultEstimateMs : 8000;
  }

  _normToolState(s) {
    s = String(s || "");
    if (s === "running") return "running";
    if (s === "error" || s === "failed") return "error";
    if (s === "success" || s === "complete" || s === "done") return "success";
    return "running";
  }

  _calcFakeProgress(dom) {
    const est = Math.max(1, Number(dom._fakeEstimateMs || 8000));
    const startAt = Number(dom._fakeStartAt || Date.now());
    const cap = (this._toolUi && this._toolUi.capRunningProgress) ? this._toolUi.capRunningProgress : 0.99;

    const elapsed = Math.max(0, Date.now() - startAt);
    const raw = elapsed / est;

    // 慢了就停 99%
    const p = Math.min(Math.max(raw, 0), cap);

    dom._fakeProgress = p;
    return p;
  }

  _updateFakeProgress(dom) {
    if (!dom || !dom.data) return;
    if (this._normToolState(dom.data.state) !== "running") return;

    const p = this._calcFakeProgress(dom);

    if (dom.fill) dom.fill.style.width = `${Math.round(p * 100)}%`;

    // 百分比：最多显示 99%
    const pct = Math.min(99, Math.max(0, Math.floor(p * 100)));
    if (dom.pctEl) dom.pctEl.textContent = `${pct}%`;
  }

  _ensureFakeProgress(dom, { server, name, progress } = {}) {
    if (!dom) return;

    dom._fakeEstimateMs = this._toolEstimateMs(server, name);

    const cap = (this._toolUi && this._toolUi.capRunningProgress) ? this._toolUi.capRunningProgress : 0.99;
    const init = Math.min(Math.max(Number(progress) || 0, 0), cap);

    if (!Number.isFinite(dom._fakeInitProgress)) dom._fakeInitProgress = init;
    else dom._fakeInitProgress = Math.max(dom._fakeInitProgress, init);

    if (!Number.isFinite(dom._fakeStartAt)) dom._fakeStartAt = NaN;

    this._updateFakeProgress(dom);
    if (dom._fakeTimer) return;

    if (dom._fakeDelayTimer) return;

    const tickMs = (this._toolUi && this._toolUi.tickMs) ? this._toolUi.tickMs : 120;
    const delayMs = (this._toolUi && Number.isFinite(this._toolUi.fakeDelayMs))
      ? Math.max(0, Number(this._toolUi.fakeDelayMs))
      : 2000;

    dom._fakeDelayTimer = setTimeout(() => {
      dom._fakeDelayTimer = null;

      if (!dom || !dom.data) return;

      const st = this._normToolState(dom.data.state);
      if (st !== "running") return;

      if (dom._progressMode === "real") return;

      if (dom._fakeTimer) return;

      const init2 = Math.min(Math.max(Number(dom._fakeInitProgress) || 0, 0), cap);
      dom._fakeStartAt = Date.now() - init2 * dom._fakeEstimateMs;
      this._updateFakeProgress(dom);

      dom._fakeTimer = setInterval(() => {
        if (!dom || !dom.data) {
          if (dom && dom._fakeTimer) clearInterval(dom._fakeTimer);
          if (dom) dom._fakeTimer = null;
          return;
        }

        const st2 = this._normToolState(dom.data.state);
        if (st2 !== "running") {
          if (dom._fakeTimer) clearInterval(dom._fakeTimer);
          dom._fakeTimer = null;
          return;
        }

        if (dom._progressMode === "real") {
          if (dom._fakeTimer) clearInterval(dom._fakeTimer);
          dom._fakeTimer = null;
          return;
        }

        this._updateFakeProgress(dom);
      }, tickMs);
    }, delayMs);
  }

  _stopFakeProgress(dom) {
    if (!dom) return;

    if (dom._fakeDelayTimer) {
      clearTimeout(dom._fakeDelayTimer);
      dom._fakeDelayTimer = null;
    }

    if (dom._fakeTimer) {
      clearInterval(dom._fakeTimer);
      dom._fakeTimer = null;
    }

    dom._fakeStartAt = NaN;
    dom._fakeProgress = 0;
    dom._fakeInitProgress = NaN;
  }

  _summaryToObject(summary) {
    if (summary == null) return null;
    if (typeof summary === "object") return summary;

    if (typeof summary === "string") {
      // 后端可能把 summary 转成 JSON 字符串
      try {
        const obj = JSON.parse(summary);
        return (obj && typeof obj === "object") ? obj : null;
      } catch {
        return null;
      }
    }
    return null;
  }

  // tool 卡片：按 tool_call_id upsert（可折叠、极简、带状态符号）
  upsertToolCard(tool_call_id, patch) {
    const wasNearBottom = this.isNearBottom();
    const clamp01 = (n) => Math.max(0, Math.min(1, Number.isFinite(n) ? n : 0));
    const safeStringify = (x) => {
      try { return JSON.stringify(x); } catch { return String(x ?? ""); }
    };
    const truncate = (s, n = 160) => {
      s = String(s ?? "");
      return s.length > n ? (s.slice(0, n) + "…") : s;
    };
    const normState = (s) => {
      s = String(s || "");
      if (s === "running") return "running";
      if (s === "error" || s === "failed") return "error";
      if (s === "success" || s === "complete" || s === "done") return "success";
      return "running";
    };

    let dom = this.toolDomById.get(tool_call_id);

    if (!dom) {
      const wrap = document.createElement("div");
      wrap.className = "msg assistant";

      const details = document.createElement("details");
      details.className = "tool-card";
      details.open = false; // 强制默认折叠

      const head = document.createElement("summary");
      head.className = "tool-head";

      // 单行：状态符号 + 工具名 + args 预览（ellipsis）
      const line = document.createElement("div");
      line.className = "tool-line";

      const left = document.createElement("div");
      left.className = "tool-left";

      const statusEl = document.createElement("span");
      statusEl.className = "tool-status";

      const nameEl = document.createElement("span");
      nameEl.className = "tool-name";

      left.appendChild(statusEl);
      left.appendChild(nameEl);

      const argsPreviewEl = document.createElement("div");
      argsPreviewEl.className = "tool-args-preview";

      line.appendChild(left);
      line.appendChild(argsPreviewEl);

      // 自定义短进度条 + 百分比
      const progRow = document.createElement("div");
      progRow.className = "tool-progress-row";

      const prog = document.createElement("div");
      prog.className = "tool-progress";

      const fill = document.createElement("div");
      fill.className = "tool-progress-fill";
      prog.appendChild(fill);

      const pctEl = document.createElement("span");
      pctEl.className = "tool-progress-pct";
      pctEl.textContent = "0%";

      progRow.appendChild(prog);
      progRow.appendChild(pctEl);

      head.appendChild(line);
      head.appendChild(progRow);

      // 展开内容：args + summary
      const bodyWrap = document.createElement("div");
      bodyWrap.className = "tool-body-wrap";

      const pre = document.createElement("pre");
      pre.className = "tool-body";

      const preview = document.createElement("div");
      preview.className = "tool-preview";
      preview.style.display = "none"; // 永久隐藏：不在 tool-card 内展示媒体

      bodyWrap.appendChild(pre);
      bodyWrap.appendChild(preview);

      details.appendChild(head);
      details.appendChild(bodyWrap);

      wrap.appendChild(details);
      this.chatEl.appendChild(wrap);
      this.maybeAutoScroll(wasNearBottom, { behavior: "auto" });

      dom = {
        wrap, details, statusEl, nameEl, argsPreviewEl, progRow, prog, fill, pctEl, pre, preview,
        data: { server: "", name: "", args: undefined, message: "", summary: null, state: "running", progress: 0 },
        _progressMode: "fake",
      };
      this.toolDomById.set(tool_call_id, dom);
    }
  
    // merge patch -> dom.data（关键：progress/end 不传 args 时要保留 start 的 args）
    const d = dom.data || {};
    const merged = {
      server: (patch && patch.server != null) ? patch.server : d.server,
      name:   (patch && patch.name != null)   ? patch.name   : d.name,
      state:  (patch && patch.state != null)  ? patch.state  : d.state,
      progress: (patch && typeof patch.progress === "number") ? patch.progress : d.progress,
      message: (patch && Object.prototype.hasOwnProperty.call(patch, "message")) ? (patch.message || "") : d.message,
      summary: (patch && Object.prototype.hasOwnProperty.call(patch, "summary")) ? patch.summary : d.summary,
      args:    (patch && Object.prototype.hasOwnProperty.call(patch, "args")) ? patch.args : d.args,
    };
    dom.data = merged;

    if (patch && patch.__progress_mode === "real") {
      dom._progressMode = "real";
    }

    const st = this._normToolState(merged.state);

    const displayName = this._toolDisplayName(merged.server, merged.name);
    dom.nameEl.textContent = displayName;

    // 状态符号
    dom.statusEl.classList.remove("is-running", "is-success", "is-error");
    if (st === "running") {
      dom.statusEl.textContent = "";
      dom.statusEl.classList.add("is-running");
    } else if (st === "success") {
      dom.statusEl.textContent = "✓";
      dom.statusEl.classList.add("is-success");
    } else {
      dom.statusEl.textContent = "!";
      dom.statusEl.classList.add("is-error");
    }

    // args 预览（单行）
    dom.argsPreviewEl.style.display = "none";
    dom.argsPreviewEl.textContent = "";

    if (st === "running") {
      dom.progRow.style.display = "flex";

      if (merged.message) {
        dom.argsPreviewEl.style.display = "block";
        dom.argsPreviewEl.textContent = merged.message;
      } else {
        dom.argsPreviewEl.style.display = "none";
        dom.argsPreviewEl.textContent = "";
      }

      if (dom._progressMode === "real") {
        this._stopFakeProgress(dom);

        const p = clamp01(merged.progress);
        if (dom.fill) dom.fill.style.width = `${Math.round(p * 100)}%`;
        if (dom.pctEl) dom.pctEl.textContent = `${Math.round(p * 100)}%`;
      } else {
        this._ensureFakeProgress(dom, {
          server: merged.server,
          name: merged.name,
          progress: merged.progress,
        });
        this._updateFakeProgress(dom);
      }
    } else {
      this._stopFakeProgress(dom);

      dom.argsPreviewEl.style.display = "none";
      dom.argsPreviewEl.textContent = "";

      dom.progRow.style.display = "none";
      dom.fill.style.width = "0%";
      if (dom.pctEl) dom.pctEl.textContent = "0%";
    }


    // 展开体内容（完整展示参数/消息/结果摘要）
    const lines = [];
    if (merged.args != null) lines.push(`args = ${JSON.stringify(merged.args, null, 2)}`);
    if (merged.message) lines.push(`message: ${merged.message}`);
    if (merged.summary != null) {
      // 把“可见的 \n”解码成真实换行
      const unescapeVisible = (s) => {
        if (typeof s !== "string") return s;
        return s
          .replace(/\\r\\n/g, "\n")
          .replace(/\\n/g, "\n")
          .replace(/\\r/g, "\r")
          .replace(/\\t/g, "\t");
      };

      let obj = merged.summary;
      if (typeof obj === "string") {
        try { obj = JSON.parse(obj); }
        catch { obj = null; }
      }

      let v = (obj && typeof obj === "object") ? obj["INFO_USER"] : undefined;

      if (typeof v === "string") {
        v = unescapeVisible(v);

        const t = v.trim();
        if ((t.startsWith("{") && t.endsWith("}")) || (t.startsWith("[") && t.endsWith("]"))) {
          try { v = JSON.stringify(JSON.parse(t), null, 2); } catch {}
        }
        lines.push(`\n${v}`);
      } else if (v != null) {
        lines.push(`${JSON.stringify(v, null, 2)}`);
      } else {
        lines.push(``);
      }
    }


    dom.pre.textContent = lines.join("\n\n").trim();

    if (merged && merged.summary != null) {
      this._upsertToolMediaMessage(tool_call_id, merged, dom);
    } else {
      // 没 summary 就清理对应媒体块（通常发生在 running/progress 阶段）
      this._removeToolMediaMessage(tool_call_id);
    }
  }

  // 语言切换时：把已存在的 tool 卡片标题也刷新
  rerenderToolCards() {
    if (!this.toolDomById) return;

    for (const [, dom] of this.toolDomById) {
      const d = dom?.data || {};
      if (dom?.nameEl) {
        dom.nameEl.textContent = this._toolDisplayName(d.server, d.name);
      }
    }
  }

  appendDevSummary(tool_call_id, { server, name, summary, is_error } = {}) {
    // 只有 developer mode 才输出
    if (!document.body.classList.contains("dev-mode")) return;
    if (!this.devLogEl) return;
    if (!tool_call_id) return;

    const fullName = `${server || ""}.${name || ""}`.replace(/^\./, "") || "MCP Tool";
    const headText = `${fullName} (${tool_call_id})${is_error ? " [error]" : ""}`;

    let summaryText = "";
    if (summary == null) {
      summaryText = "（无 summary）";
    } else if (typeof summary === "string") {
      summaryText = summary;
    } else {
      try { summaryText = JSON.stringify(summary, null, 2); }
      catch { summaryText = String(summary); }
    }

    let dom = this.devDomByID.get(tool_call_id);
    if (!dom) {
      const item = document.createElement("div");
      item.className = "devlog-item";

      const head = document.createElement("div");
      head.className = "devlog-head";
      head.textContent = headText;

      const pre = document.createElement("pre");
      pre.className = "devlog-pre";
      pre.textContent = summaryText;

      item.appendChild(head);
      item.appendChild(pre);

      this.devLogEl.appendChild(item);
      this.devDomByID.set(tool_call_id, { item, head, pre });
    } else {
      dom.head.textContent = headText;
      dom.pre.textContent = summaryText;
    }

    // 自动滚到底部，便于实时追踪
    requestAnimationFrame(() => {
      const el = this.devLogEl;
      if (!el) return;
      el.scrollTop = el.scrollHeight;
    });
  }

  // 工具调用结果中展示视频、图片、音频
  _stripUrlQueryHash(u) {
    return String(u ?? "").split("#")[0].split("?")[0];
  }

  _basenameFromUrl(u) {
    const s = this._stripUrlQueryHash(u);
    const parts = s.split(/[\\/]/);
    return parts[parts.length - 1] || s;
  }

  _guessMediaKindFromUrl(u) {
    const s = this._stripUrlQueryHash(u).toLowerCase();
    const m = s.match(/\.([a-z0-9]+)$/);
    const ext = m ? "." + m[1] : "";

    if ([".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"].includes(ext)) return "image";
    if ([".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"].includes(ext)) return "video";
    if ([".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"].includes(ext)) return "audio";
    return "unknown";
  }

  _isSafeMediaUrl(u) {
    const s = String(u ?? "").trim();
    if (!s) return false;
    try {
      const parsed = new URL(s, window.location.href);
      const proto = String(parsed.protocol || "").toLowerCase();
      // allow: same-origin relative -> becomes http(s) here; allow absolute http(s) and blob
      return proto === "http:" || proto === "https:" || proto === "blob:";
    } catch {
      return false;
    }
  }

  _getPreviewUrlsFromSummary(summary) {
    let obj = summary;
    if (typeof obj === "string") {
      try { obj = JSON.parse(obj); } catch { return []; }
    }
    const urls = obj && obj.preview_urls;
    if (!Array.isArray(urls)) return [];
    return urls.filter((u) => typeof u === "string" && u.trim());
  }

  _extractMediaItemsFromSummary(summary) {
    const raws = this._getPreviewUrlsFromSummary(summary);
    const out = [];
    const seen = new Set();

    for (const raw of raws) {
      const url = this._normalizePreviewUrl(raw);
      if (!url) continue;

      // 关键：kind 用 raw 判定（因为 /preview?path=... 本身不带后缀）
      const kind = this._guessMediaKindFromUrl(String(raw));
      if (kind === "unknown") continue;

      const key = this._stripUrlQueryHash(String(raw));
      if (seen.has(key)) continue;
      seen.add(key);

      out.push({
        url,                               // 可访问 URL：网络/或 /api/.../preview?path=...
        kind,
        name: this._basenameFromUrl(String(raw)),
      });
    }

    return out;
  }

  _makeToolPreviewTitle(text) {
    const t = document.createElement("div");
    t.className = "tool-preview-title";
    t.textContent = String(text ?? "");
    return t;
  }

  _makeInlineVideoBlock(item, title) {
    const block = document.createElement("div");
    block.className = "tool-preview-block";

    if (title) block.appendChild(this._makeToolPreviewTitle(title));

    const v = document.createElement("video");
    v.style.objectFit = "contain";
    v.style.objectPosition = "center";
    v.className = "tool-inline-video";
    v.controls = true;
    v.preload = "metadata";
    v.playsInline = true;
    v.src = item.url;
    block.appendChild(v);

    const actions = document.createElement("div");
    actions.className = "tool-preview-actions";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tool-preview-btn";
    btn.textContent = __t("tool.preview.btn_modal");
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.openPreview({ kind: "video", file_url: item.url, name: item.name });
    });
    actions.appendChild(btn);

    const link = document.createElement("a");
    link.className = "tool-preview-link";
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = __t("tool.preview.btn_open");
    actions.appendChild(link);

    block.appendChild(actions);

    return block;
  }

  _makeAudioListBlock(items, title, { maxItems = AUDIO_PREVIEW_MAX } = {}) {
    const block = document.createElement("div");
    block.className = "tool-preview-block";

    if (title) block.appendChild(this._makeToolPreviewTitle(title));

    const list = document.createElement("div");
    list.className = "tool-audio-list";

    const show = items.slice(0, maxItems);
    show.forEach((it, idx) => {
      const row = document.createElement("div");
      row.className = "tool-audio-item";

      const label = document.createElement("div");
      label.className = "tool-media-label";
      label.textContent = it.name || __t("tool.preview.label.audio", { i: idx + 1 });
      row.appendChild(label);

      const a = document.createElement("audio");
      a.controls = true;
      a.preload = "metadata";
      a.src = it.url;
      row.appendChild(a);

      list.appendChild(row);
    });

    block.appendChild(list);

    if (items.length > maxItems) {
      const more = document.createElement("div");
      more.className = "tool-media-more";
      more.textContent = __t("tool.preview.more_audios", { n: items.length - maxItems });
      block.appendChild(more);
    }

    return block;
  }

  _makeMediaGridBlock(items, { title, kind, labelKey, maxItems = 12 } = {}) {
    const block = document.createElement("div");
    block.className = "tool-preview-block";

    if (title) block.appendChild(this._makeToolPreviewTitle(title));

    const grid = document.createElement("div");
    grid.className = "tool-media-grid";

    // 根据宽高给 thumb 打标签，动态调整 aspect-ratio
    const applyThumbAspect = (thumb, w, h) => {
      const W = Number(w) || 0;
      const H = Number(h) || 0;
      if (!(W > 0 && H > 0)) return;

      thumb.classList.remove("is-portrait", "is-square");
      const r = W / H;

      // square: 0.92~1.08
      if (r >= 0.92 && r <= 1.08) {
        thumb.classList.add("is-square");
        return;
      }
      // portrait: r < 1
      if (r < 1) {
        thumb.classList.add("is-portrait");
      }
    };

    const show = items.slice(0, maxItems);
    show.forEach((it, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tool-media-item";
      btn.title = it.name || it.url;

      const thumb = document.createElement("div");
      thumb.className = "tool-media-thumb";

      if (kind === "image") {
        const img = document.createElement("img");
        img.src = it.url;
        img.alt = it.name || "";

        // FIX(1): 强制不裁切（不依赖 CSS 是否命中/是否被覆盖）
        img.style.objectFit = "contain";
        img.style.objectPosition = "center";

        img.addEventListener("load", () => {
          applyThumbAspect(thumb, img.naturalWidth, img.naturalHeight);
        });

        thumb.appendChild(img);
      } else if (kind === "video") {
        const v = document.createElement("video");
        v.preload = "metadata";
        v.muted = true;
        v.playsInline = true;

        // FIX(1): 强制不裁切
        v.style.objectFit = "contain";
        v.style.objectPosition = "center";

        const apply = () => applyThumbAspect(thumb, v.videoWidth, v.videoHeight);
        // 先绑定，再设置 src，避免缓存命中导致事件丢失
        v.addEventListener("loadedmetadata", apply, { once: true });
        // 少数浏览器/资源场景 loadedmetadata 不稳定，再用 loadeddata 兜底一次
        v.addEventListener("loadeddata", apply, { once: true });

        v.src = it.url;

        thumb.appendChild(v);
        if (v.readyState >= 1) apply();

        const play = document.createElement("div");
        play.className = "tool-media-play";
        thumb.appendChild(play);
      }

      btn.appendChild(thumb);

      const label = document.createElement("div");
      label.className = "tool-media-label";
      const fallbackKey =
        labelKey ||
        (kind === "video" ? "tool.preview.label.video" : "tool.preview.label.image");

      label.textContent = it.name || __t(fallbackKey, { i: idx + 1 });
      btn.appendChild(label);

      btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.openPreview({ kind, file_url: it.url, name: it.name });
      });

      grid.appendChild(btn);
    });

    block.appendChild(grid);

    if (items.length > maxItems) {
      const more = document.createElement("div");
      more.className = "tool-media-more";
      more.textContent = __t("tool.preview.more_items", { n: items.length - maxItems });
      block.appendChild(more);
    }

    return block;
  }

  _removeToolMediaMessage(tool_call_id) {
    const dom = this.toolMediaDomById && this.toolMediaDomById.get(tool_call_id);
    if (dom) {
      try { dom.wrap?.remove(); } catch {}
      this.toolMediaDomById.delete(tool_call_id);
    }
  }

  // 在 chat 列表中，把“媒体预览块”插在 tool-card 后面（不放进 tool-card）
  _upsertToolMediaMessage(tool_call_id, merged, toolCardDom) {
    if (!tool_call_id) return;

    const summary = merged?.summary;
    if (summary == null) {
      // 没 summary 就不展示（也可选择清理旧的）
      this._removeToolMediaMessage(tool_call_id);
      return;
    }

    // 从 summary.preview_urls 提取媒体
    const media = this._extractMediaItemsFromSummary(summary);
    if (!media || !media.length) {
      this._removeToolMediaMessage(tool_call_id);
      return;
    }

    // 已存在就复用（并确保位置在 tool-card 之后）
    let dom = this.toolMediaDomById.get(tool_call_id);

    const wasNearBottom = this.isNearBottom();

    if (!dom) {
      const wrap = document.createElement("div");
      wrap.className = "msg assistant tool-media-msg";

      const card = document.createElement("div");
      card.className = "media-card";

      const preview = document.createElement("div");
      // 复用现有 tool-preview 的样式与内部 block 结构
      preview.className = "tool-preview";

      card.appendChild(preview);
      wrap.appendChild(card);

      // 插入到 tool-card 之后（保证顺序：tool card -> media）
      if (toolCardDom && toolCardDom.wrap && toolCardDom.wrap.parentNode) {
        toolCardDom.wrap.after(wrap);
      } else {
        this.chatEl.appendChild(wrap);
      }

      dom = { wrap, card, preview };
      this.toolMediaDomById.set(tool_call_id, dom);

      this.maybeAutoScroll(wasNearBottom, { behavior: "auto" });
    } else {
      // 如果 DOM 顺序被打乱，强制挪回 tool-card 后面
      try {
        if (toolCardDom && toolCardDom.wrap && dom.wrap && toolCardDom.wrap.nextSibling !== dom.wrap) {
          toolCardDom.wrap.after(dom.wrap);
        }
      } catch {}
    }

    this._renderToolMediaPreview({ preview: dom.preview, details: null }, merged);
  }


  _renderToolMediaPreview(dom, merged) {
    if (!dom || !dom.preview) return;

    const st = this._normToolState(merged?.state);
    const summary = merged?.summary;

    // running 且无 summary：清空，避免复用上一轮残留
    if (st === "running" && summary == null) {
      dom.preview.innerHTML = "";
      dom.preview._lastMediaKey = "";
      return;
    }

    if (summary == null) {
      dom.preview.innerHTML = "";
      dom.preview._lastMediaKey = "";
      return;
    }

    const lang = __osNormLang(window.OPENSTORYLINE_LANG || "zh");

    let key = "";
    try {
      key = (typeof summary === "string") ? summary : JSON.stringify(summary);
    } catch {
      key = String(summary);
    }

    const combinedKey = `${lang}::${key}`;
    if (dom.preview._lastMediaKey === combinedKey) return;
    dom.preview._lastMediaKey = combinedKey;

    const media = this._extractMediaItemsFromSummary(summary);
    if (!media.length) {
      dom.preview.innerHTML = "";
      return;
    }

    const toolName = String(merged?.name || "").toLowerCase();
    const toolFull = String(this._toolFullName(merged?.server, merged?.name) || "").toLowerCase();

    const isSplitShots = toolName.includes("split_shots") || toolFull.includes("split_shots");
    const isRender = toolName.includes("render") || toolFull.includes("render");
    const isTtsOrMusic =
      toolName.includes("tts") || toolFull.includes("tts") ||
      toolName.includes("music") || toolFull.includes("music");

    const videos = media.filter((x) => x.kind === "video");
    const audios = media.filter((x) => x.kind === "audio");
    const images = media.filter((x) => x.kind === "image");

    dom.preview.innerHTML = "";

    // Render：成片直接内嵌展示（第一条 video）
    if (isRender && videos.length) {
      dom.preview.appendChild(this._makeInlineVideoBlock(videos[0], __t("tool.preview.render_title")));

      const restVideos = videos.slice(1);
      if (restVideos.length) {
        dom.preview.appendChild(this._makeMediaGridBlock(restVideos, {
          title: __t("tool.preview.other_videos"),
          kind: "video",
          labelKey: "tool.preview.label.video",
          maxItems: 8,
        }));
      }

      if (audios.length) {
        dom.preview.appendChild(this._makeAudioListBlock(audios, __t("tool.preview.audio")));
      }

      if (images.length) {
        dom.preview.appendChild(this._makeMediaGridBlock(images, {
          title: __t("tool.preview.images"),
          kind: "image",
          labelKey: "tool.preview.label.image",
          maxItems: 12,
        }));
      }

      // 关键节点：完成后默认展开，做到“直接展示成片”
      if (st !== "running" && dom.details) dom.details.open = true;
      return;
    }

    // 配音/音乐：优先展示试听
    if (isTtsOrMusic && audios.length) {
      dom.preview.appendChild(this._makeAudioListBlock(audios, __t("tool.preview.listen")));
      if (st !== "running" && dom.details) dom.details.open = true;
    }

    // 镜头切分：展示切分后视频（可点击弹窗预览）
    if (videos.length) {
      dom.preview.appendChild(this._makeMediaGridBlock(videos, {
        title: isSplitShots ? __t("tool.preview.split_shots") : __t("tool.preview.videos"),
        kind: "video",
        labelKey: isSplitShots ? "tool.preview.label.shot" : "tool.preview.label.video",
        maxItems: isSplitShots ? 12 : 8,
      }));
      if (isSplitShots && st !== "running" && dom.details) dom.details.open = true;
    }

    // 图片
    if (images.length) {
      dom.preview.appendChild(this._makeMediaGridBlock(images, {
        title: __t("tool.preview.images"),
        kind: "image",
        labelKey: "tool.preview.label.image",
        maxItems: 12,
      }));
    }

    // 其它工具也可能产生音频：给一个通用展示
    if (!isTtsOrMusic && audios.length) {
      dom.preview.appendChild(this._makeAudioListBlock(audios, __t("tool.preview.audio")));
    }
  }

  _isLikelyLocalPath(s) {
    s = String(s ?? "").trim();
    if (!s) return false;
    // 相对路径：.xxx 或 xxx/yyy；绝对路径：/xxx/yyy
    if (s.startsWith(".") || s.startsWith("/")) return true;
    // windows 盘符（可选兜底）
    if (/^[a-zA-Z]:[\\/]/.test(s)) return true;
    return false;
  }

  

  // 只认为“显式 scheme”的才是网络 URL，避免把 .server_cache/... 误判成 http(s) 相对 URL
  _isAbsoluteNetworkUrl(s) {
    s = String(s ?? "").trim().toLowerCase();
    return s.startsWith("http://") || s.startsWith("https://") || s.startsWith("blob:");
  }

  // 已经是你服务端可直接访问的相对路径（不要再走 preview 代理）
  _isServedRelativeUrlPath(s) {
    s = String(s ?? "").trim();
    return s.startsWith("/api/") || s.startsWith("/static/");
  }

  // 判断“服务器本地路径”
  // - .server_cache/..
  // - ./xxx/..
  // - /abs/path/.. （但排除 /api/, /static/）
  // - windows: C:\...
  // - 其它不带 scheme 且包含 / 或 \ 的相对路径（例如 outputs/xxx.mp4）
  _isLikelyServerLocalPath(s) {
    s = String(s ?? "").trim();
    if (!s) return false;

    if (this._isServedRelativeUrlPath(s)) return false; // 已可访问

    if (/^[a-zA-Z]:[\\/]/.test(s)) return true; // Windows drive
    if (s.startsWith(".") || s.startsWith("./") || s.startsWith(".\\")) return true;

    if (s.startsWith("/")) return true; // 绝对路径（同样排除 /api,/static 已在上面处理）

    // 没 scheme，但像路径（含斜杠）
    if (!this._isAbsoluteNetworkUrl(s) && (s.includes("/") || s.includes("\\"))) return true;

    return false;
  }

  _localPathToPreviewUrl(p) {
    const sid = this._sessionId;
    if (!sid) return null;
    return `/api/sessions/${encodeURIComponent(sid)}/preview?path=${encodeURIComponent(String(p ?? ""))}`;
  }

  // 将 preview_urls 里的 raw 字符串转为真正可在浏览器加载的 URL
  _normalizePreviewUrl(raw) {
    const s = String(raw ?? "").trim();
    if (!s) return null;

    // 1) 已可访问的相对 URL
    if (this._isServedRelativeUrlPath(s)) return s;

    // 2) 显式网络 URL
    if (this._isAbsoluteNetworkUrl(s)) return s;

    // 3) 本地路径 -> preview 代理
    if (this._isLikelyServerLocalPath(s)) return this._localPathToPreviewUrl(s);

    return null;
  }


  openPreview(media) {
    if (!this._modalBound) this.bindModalClose();

    this.modalContent.innerHTML = "";
    this.modalEl.classList.remove("hidden");

    const preferSrc = media.local_url || media.file_url;

    if (media.kind === "image") {
      const img = document.createElement("img");
      img.src = preferSrc;
      img.alt = media.name || "";
      this.modalContent.appendChild(img);
      return;
    }

    if (media.kind === "video") {
      const v = document.createElement("video");
      v.src = preferSrc;
      v.controls = true;
      v.autoplay = true;
      v.preload = "metadata";
      this.modalContent.appendChild(v);
      return;
    }

    if (media.kind === "audio") {
      const a = document.createElement("audio");
      a.src = preferSrc;
      a.controls = true;
      a.autoplay = true;
      a.preload = "metadata";
      this.modalContent.appendChild(a);
      return;
    }

    const box = document.createElement("div");
    box.className = "file-fallback";

    const pad = document.createElement("div");
    pad.style.padding = "16px";

    const tip = document.createElement("div");
    tip.style.color = "rgba(0,0,0,0.75)";
    tip.style.marginBottom = "8px";
    tip.textContent = __t("preview.unsupported");
    pad.appendChild(tip);

    const name = document.createElement("div");
    name.style.fontFamily = "ui-monospace,monospace";
    name.style.fontSize = "12px";
    name.style.marginBottom = "12px";
    name.textContent = media.name || media.id || "";
    pad.appendChild(name);

    const link = document.createElement("a");
    link.href = media.file_url || preferSrc || "#";
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = __t("preview.open_download");
    pad.appendChild(link);

    box.appendChild(pad);
    this.modalContent.appendChild(box);
  }

  closePreview() {
    this.modalEl.classList.add("hidden");
    this.modalContent.innerHTML = "";
  }

  rerenderToolMediaPreviews() {
    if (!this.toolMediaDomById) return;

    for (const [tool_call_id, mediaDom] of this.toolMediaDomById) {
      const toolDom = this.toolDomById && this.toolDomById.get(tool_call_id);
      const merged = toolDom && toolDom.data;
      if (!mediaDom || !mediaDom.preview || !merged) continue;

      this._renderToolMediaPreview({ preview: mediaDom.preview, details: null }, merged);
    }
  }


  bindModalClose() {
    // 防止重复绑定（openPreview 里也会兜底调用一次）
    if (this._modalBound) return;
    this._modalBound = true;

    const close = (e) => {
      if (e) {
        e.preventDefault();
        e.stopPropagation();
        // 同一元素上其它监听也停掉，避免“关闭后又被底层点击重新打开”
        if (typeof e.stopImmediatePropagation === "function") e.stopImmediatePropagation();
      }
      this.closePreview();
    };

    // 1) 明确绑定 backdrop/close
    if (this.modalBackdrop) {
      this.modalBackdrop.addEventListener("click", close, true); // capture
      this.modalBackdrop.addEventListener("pointerdown", close, true); // 兼容移动端/某些浏览器
    }
    if (this.modalClose) {
      this.modalClose.addEventListener("click", close, true);
      this.modalClose.addEventListener("pointerdown", close, true);
    }

    // 2) 兜底：document capture 里判断“点到内容区外”就关闭
    document.addEventListener("click", (e) => {
      if (!this.modalEl || this.modalEl.classList.contains("hidden")) return;

      const t = e.target;

      // 点到 close（或其子元素） => 关闭
      if (this.modalClose && (t === this.modalClose || this.modalClose.contains(t))) {
        close(e);
        return;
      }

      // 点到内容区内部 => 不关闭（允许操作 video controls/滚动等）
      if (this.modalContent && (t === this.modalContent || this.modalContent.contains(t))) {
        return;
      }

      // 其他任何地方（含 click 穿透到页面底层）=> 关闭
      close(e);
    }, true);

    // 3) Esc 关闭
    document.addEventListener("keydown", (e) => {
      if (!this.modalEl || this.modalEl.classList.contains("hidden")) return;
      if (e.key === "Escape") {
        e.preventDefault();
        this.closePreview();
      }
    }, true);
  }


  escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
    }[c]));
  }
}

class App {
  constructor() {
    this.api = new ApiClient();
    this.ui = new ChatUI();
    this.ws = null;

    this.sessionId = null;
    this.sessionHistory = [];
    this.pendingMedia = [];

    this.llmSelect = $("#llmModelSelect");
    this.vlmSelect = $("#vlmModelSelect");

    this.llmModels = [];
    this.vlmModels = [];

    this.llmModel = null;
    this.vlmModel = null;

    // custom model section
    this.customLlmSection = $("#customLlmSection");
    this.customVlmSection = $("#customVlmSection");

    // Custom model UI
    this.customLlmModel = $("#customLlmModel");
    this.customLlmBaseUrl = $("#customLlmBaseUrl");
    this.customLlmApiKey = $("#customLlmApiKey");
    this.customVlmModel = $("#customVlmModel");
    this.customVlmBaseUrl = $("#customVlmBaseUrl");
    this.customVlmApiKey = $("#customVlmApiKey");

    // TTS UI
    this.ttsBox = $("#ttsBox");
    this.ttsProviderSelect = $("#ttsProviderSelect");
    this.ttsProviderFieldsHost = $("#ttsProviderFields");
    this.ttsUiSchema = null;

    // Pexels UI
    this.pexelsBox = $("#pexelsBox");
    this.pexelsKeyModeSelect = $("#pexelsKeyModeSelect");
    this.pexelsCustomKeyBox = $("#pexelsCustomKeyBox");
    this.pexelsApiKeyInput = $("#pexelsApiKeyInput");

    this.limits = {
      max_media_per_session: 30,
      max_pending_media_per_session: 30,
      upload_chunk_bytes: 8 * 1024 * 1024,
    };

    this.localObjectUrlByMediaId = new Map();

    this.fileInput = $("#fileInput");
    this.uploadBtn = $("#uploadBtn");
    this.promptInput = $("#promptInput");
    this.sendBtn = $("#sendBtn");
    this.quickPromptBtn = $("#quickPromptBtn");
    this._quickPromptIdx = 0;
    this.sidebarToggleBtn = $("#sidebarToggle");
    this.createDialogBtn = $("#createDialogBtn");
    this.devbarToggleBtn = $("#devbarToggle");
    this.devbarEl = $("#devbar");
    this.sessionHistoryListEl = $("#sessionHistoryList");

    this.canceling = false;

    // 保存“发送箭头”的原始 SVG
    this._sendIconSend = this.sendBtn ? this.sendBtn.innerHTML : "";

    // “打断”图标：白色实心正方形
    this._sendIconStop = `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="5" y="5" width="14" height="14" rx="1.2" fill="currentColor" stroke="none"></rect>
      </svg>
    `;

    this.streaming = false;
    this.uploading = false;

    // 防止 “发送后到 assistant.start 前” 的短暂切换窗口
    this._switchLockUntilAssistantStart = false;
    this._switchLockTimer = 0;

    this.langToggle = $("#langToggle");
    this.lang = __osNormLang(window.OPENSTORYLINE_LANG || "zh");

    this._langWasStored = (__osLoadLang() != null);

  }

  _showSwitchWhileStreamingToast() {
    if (!this.ui || !this.ui.showToastI18n) return;
    this.ui.showToastI18n("toast.switch_while_streaming", {});
    setTimeout(() => this.ui.hideToast(), 1800);
  }

  _isSwitchBlocked() {
    return !!this.streaming || !!this._switchLockUntilAssistantStart;
  }

  _setSwitchLock() {
    this._switchLockUntilAssistantStart = true;
    if (this._switchLockTimer) {
      try { clearTimeout(this._switchLockTimer); } catch {}
      this._switchLockTimer = 0;
    }

    // 极端情况下（网络抖动/WS 事件丢失）assistant.start 可能迟迟不到：
    // 给一个超时自动释放，避免“永久无法切换会话”。
    this._switchLockTimer = setTimeout(() => {
      this._switchLockTimer = 0;
      this._switchLockUntilAssistantStart = false;
    }, 25000);
  }

  _clearSwitchLock() {
    this._switchLockUntilAssistantStart = false;
    if (this._switchLockTimer) {
      try { clearTimeout(this._switchLockTimer); } catch {}
      this._switchLockTimer = 0;
    }
  }

  wsUrl(sessionId) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/ws/sessions/${encodeURIComponent(sessionId)}/chat`;
  }

  async bootstrap() {
    // this.restoreSidebarState();
    // this.restoreDevbarState();
    this.ui.bindModalClose();
    this.bindUI();
    this._setLang(this.lang, { persist: false, syncServer: false });
    await this.loadTtsUiSchema();

    // 先加载本地会话列表
    this.sessionHistory = this._loadSessionHistory();
    this._renderSessionHistory();

    // 复用 localStorage 当前会话；如果失效就创建新 session
    const saved = localStorage.getItem(SESSION_ID_KEY);
    if (saved) {
      try {
        const snap = await this.api.getSession(saved);
        await this.useSession(saved, snap);
        return;
      } catch (err) {
        // 仅当明确 404（会话不存在）时才清理本地记录；其它错误（例如网络抖动）不要误删
        if (err && err.status === 404) {
          localStorage.removeItem(SESSION_ID_KEY);
          this._removeSessionFromHistory(saved);
          this._renderSessionHistory();
        } else {
          console.warn("[session] failed to restore saved session (non-404), keep local record:", saved, err);
        }
      }
    }

    await this.newSession();
  }

  // ----- Session history (localStorage, per-browser) -----

  _loadSessionHistory() {
    try {
      const raw = localStorage.getItem(SESSION_LIST_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];

      const out = [];
      for (const item of parsed) {
        if (!item || typeof item !== "object") continue;
        const id = String(item.id || "").trim();
        if (!id) continue;
        const title = String(item.title || "").trim();
        const created_at = Number(item.created_at || Date.now());
        const updated_at = Number(item.updated_at || created_at);
        const last_preview = String(item.last_preview || "").trim();
        const has_user = !!item.has_user;
        out.push({ id, title, created_at, updated_at, last_preview, has_user });
      }
      return out;
    } catch {
      return [];
    }
  }

  _saveSessionHistory(list) {
    if (!Array.isArray(list)) return;
    try {
      // 根据 created_at 倒序排序再裁剪，确保持久化里保留“最新的 50 个会话”，并与 UI 排序一致
      const sorted = list
        .slice()
        .sort((a, b) => (Number(b?.created_at) || 0) - (Number(a?.created_at) || 0));
      const trimmed = sorted.slice(0, 50);
      localStorage.setItem(SESSION_LIST_KEY, JSON.stringify(trimmed));
    } catch {
      // 忽略存储错误（Safari 无痕等）
    }
  }

  _newChatTitle() {
    const lang = __osNormLang(this.lang || "zh");
    return (lang === "en") ? "New chat" : "新对话";
  }

  _isNewChatTitle(title) {
    const t = String(title || "").trim();
    if (!t) return true;
    return t === "新对话" || t === "New chat";
  }

  _deriveSessionTitle(snapshot, fallbackId) {
    const history = (snapshot && Array.isArray(snapshot.history)) ? snapshot.history : [];
    const firstUser = history.find((h) => h && h.role === "user" && typeof h.content === "string");
    let title = firstUser ? String(firstUser.content || "") : "";
    title = title.replace(/\s+/g, " ").trim();
    if (title) {
      if (title.length > 32) title = title.slice(0, 32) + "…";
      return title;
    }
    // 还没输入任何内容：显示“新对话”
    return this._newChatTitle();
  }

  _deriveSessionPreview(snapshot) {
    const history = (snapshot && Array.isArray(snapshot.history)) ? snapshot.history : [];
    // 优先最后一条用户消息，其次助手
    let last = null;
    for (let i = history.length - 1; i >= 0; i--) {
      const it = history[i];
      if (!it || typeof it !== "object") continue;
      if (!last && (it.role === "assistant" || it.role === "user") && typeof it.content === "string") {
        last = it;
      }
      if (it.role === "user" && typeof it.content === "string") {
        last = it;
        break;
      }
    }
    if (!last || typeof last.content !== "string") return "";
    let text = String(last.content || "").replace(/\s+/g, " ").trim();
    if (!text) return "";
    if (text.length > 40) text = text.slice(0, 40) + "…";
    return text;
  }

  _snapshotHasUserMessage(snapshot) {
    const history = (snapshot && Array.isArray(snapshot.history)) ? snapshot.history : [];
    return history.some((h) => h && h.role === "user" && typeof h.content === "string" && String(h.content || "").trim());
  }

  _upsertSessionHistoryFromSnapshot(sessionId, snapshot) {
    const sid = String(sessionId || "").trim();
    if (!sid) return;
    if (!Array.isArray(this.sessionHistory)) this.sessionHistory = [];

    const now = Date.now();
    const title = this._deriveSessionTitle(snapshot, sid);
    const preview = this._deriveSessionPreview(snapshot);
    const hasUser = this._snapshotHasUserMessage(snapshot);

    let found = false;
    const next = this.sessionHistory.map((item) => {
      if (item.id !== sid) return item;
      found = true;
      return {
        ...item,
        title: title || item.title || "",
        last_preview: preview || item.last_preview || "",
        // updated_at 仅作元信息保存，不参与排序；更新不会影响列表顺序
        updated_at: now,
        has_user: (item.has_user || false) || hasUser,
      };
    });

    if (!found) {
      next.push({
        id: sid,
        title,
        created_at: now,
        updated_at: now,
        last_preview: preview,
        has_user: hasUser,
      });
    }

    this.sessionHistory = next;
    this._saveSessionHistory(this.sessionHistory);
  }

  _removeSessionFromHistory(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid || !Array.isArray(this.sessionHistory)) return;
    const next = this.sessionHistory.filter((item) => item.id !== sid);
    if (next.length === this.sessionHistory.length) return;
    this.sessionHistory = next;
    this._saveSessionHistory(this.sessionHistory);
  }

  _renderSessionHistory(activeId) {
    const host = this.sessionHistoryListEl || $("#sessionHistoryList");
    if (!host) return;

    host.innerHTML = "";

    const list = Array.isArray(this.sessionHistory) ? this.sessionHistory.slice() : [];
    if (!list.length) {
      const empty = document.createElement("div");
      empty.className = "session-history-empty";
      empty.textContent = __t("sidebar.history_empty");
      host.appendChild(empty);
      return;
    }

    // 始终按创建时间倒序展示，点击切换会话不会改变顺序
    list.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

    for (const item of list) {
      if (!item || !item.id) continue;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "session-history-item";
      btn.dataset.sessionId = String(item.id);
      if (activeId && String(activeId) === String(item.id)) {
        btn.classList.add("is-active");
      }

      const titleEl = document.createElement("div");
      titleEl.className = "session-history-title";
      titleEl.textContent = item.title || this._newChatTitle();
      btn.appendChild(titleEl);

      if (item.last_preview) {
        const metaEl = document.createElement("div");
        metaEl.className = "session-history-meta";
        metaEl.textContent = item.last_preview;
        btn.appendChild(metaEl);
      }

      host.appendChild(btn);
    }
  }

  _getSessionHistoryItem(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid || !Array.isArray(this.sessionHistory)) return null;
    return this.sessionHistory.find((x) => x && x.id === sid) || null;
  }

  _isCurrentSessionBlank() {
    if (!this.sessionId) return false;
    const item = this._getSessionHistoryItem(this.sessionId);
    const hasUser = !!(item && item.has_user);
    // “新对话/草稿会话”的判定：只看是否出现过用户消息。
    // 说明：上传了素材但还没发送第一条消息时，仍然应该被视为同一个“新对话”，避免无限创建多个新会话。
    return !hasUser;
  }

  _findAnyBlankSessionId() {
    if (!Array.isArray(this.sessionHistory) || !this.sessionHistory.length) return null;
    // 复用“最新的空会话”，保证“全局只保留一个新对话”的交互更符合直觉
    const blanks = this.sessionHistory
      .filter((it) => it && !it.has_user && it.id)
      .slice()
      .sort((a, b) => (Number(b.created_at) || 0) - (Number(a.created_at) || 0));
    return blanks.length ? String(blanks[0].id || "") : null;
  }

  _touchBlankSessionAsNewChat(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid) return;
    const now = Date.now();
    const next = (Array.isArray(this.sessionHistory) ? this.sessionHistory : []).map((it) => {
      if (!it || it.id !== sid) return it;
      return {
        ...it,
        title: this._newChatTitle(),
        last_preview: "",
        has_user: false,
        created_at: now, // 视为“新对话”，刷新创建时间使其成为最新
        updated_at: now,
      };
    });
    this.sessionHistory = next;
    this._saveSessionHistory(this.sessionHistory);
    this._renderSessionHistory(this.sessionId);
  }

  _handleMissingSessionOnClick(sessionId) {
    // 遇到错误时，直接从本地列表移除，并用 toast 提示
    const sid = String(sessionId || "").trim();
    if (!sid) return;

    this._removeSessionFromHistory(sid);
    this._renderSessionHistory(this.sessionId);

    const lang = __osNormLang(this.lang || "zh");
    const msg = (lang === "en")
      ? "This chat has expired or is no longer available and has been removed from history."
      : "该会话已失效或不可用，已从历史列表中移除。";
    if (this.ui && typeof this.ui.showToast === "function") {
      this.ui.showToast(msg);
      setTimeout(() => this.ui.hideToast(), 2000);
    }
  }

  _markSessionHasUserNow(userText) {
    const sid = String(this.sessionId || "").trim();
    if (!sid) return;

    const text = String(userText || "").replace(/\s+/g, " ").trim();
    const preview = text ? (text.length > 40 ? text.slice(0, 40) + "…" : text) : "";
    const now = Date.now();

    if (!Array.isArray(this.sessionHistory)) this.sessionHistory = [];

    let found = false;
    const next = this.sessionHistory.map((it) => {
      if (!it || it.id !== sid) return it;
      found = true;
      const nextTitle = (this._isNewChatTitle(it.title) || !String(it.title || "").trim())
        ? (text.length > 32 ? text.slice(0, 32) + "…" : (text || this._newChatTitle()))
        : it.title;

      return {
        ...it,
        title: nextTitle,
        last_preview: preview || it.last_preview || "",
        has_user: true,
        updated_at: now,
      };
    });

    if (!found) {
      next.push({
        id: sid,
        title: text ? (text.length > 32 ? text.slice(0, 32) + "…" : text) : this._newChatTitle(),
        created_at: now,
        updated_at: now,
        last_preview: preview,
        has_user: true,
      });
    }

    this.sessionHistory = next;
    this._saveSessionHistory(this.sessionHistory);
    this._renderSessionHistory(this.sessionId);
  }

  _flashSessionHistoryItem(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid) return;
    const host = this.sessionHistoryListEl || $("#sessionHistoryList");
    if (!host) return;

    const el = host.querySelector(`[data-session-id="${sid}"]`);
    if (!el) return;

    el.classList.add("flash");
    setTimeout(() => {
      el.classList.remove("flash");
    }, 160);
  }

  async loadTtsUiSchema() {
    let schema = null;
    try {
      schema = await this.api.getTtsUiSchema();
    } catch (e) {
      console.warn("[tts] failed to load /api/meta/tts:", e);
    }

    this.ttsUiSchema = schema;
    this._renderTtsUiFromSchema(schema);
  }

  _renderTtsUiFromSchema(schema) {
    if (!this.ttsProviderSelect || !this.ttsProviderFieldsHost) return;

    const providers = (schema && Array.isArray(schema.providers)) ? schema.providers : [];
    const before = String(this.ttsProviderSelect.value || "").trim();

    this.ttsProviderSelect.innerHTML = "";
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = __t("sidebar.tts_default");
    this.ttsProviderSelect.appendChild(opt0);

    for (const v of providers) {
      const provider = String(v?.provider || "").trim();
      if (!provider) continue;
      const label = String(v?.label || provider);

      const opt = document.createElement("option");
      opt.value = provider;
      opt.textContent = label;
      this.ttsProviderSelect.appendChild(opt);
    }

    this.ttsProviderFieldsHost.innerHTML = "";

    for (const v of providers) {
      const provider = String(v?.provider || "").trim();
      if (!provider) continue;

      const block = document.createElement("div");
      block.className = "sidebar-tts-fields hidden";
      block.dataset.ttsProvider = provider;

      const fields = Array.isArray(v?.fields) ? v.fields : [];

      for (const f of fields) {
        const key = String(f?.key || "").trim();
        if (!key) continue;

        const label = String(f?.label || key).trim();

        const required = !!f?.required;
        const secret = !!f?.secret;

        const input = document.createElement("input");
        input.className = "sidebar-input";
        input.type = secret ? "password" : "text";
        input.autocomplete = "off";
        const basePh = String(f?.placeholder || label).trim();
        const needSuffix = !f?.placeholder; // 仅当 schema 没给 placeholder 时，加“留空使用默认”的 suffix

        input.setAttribute("data-os-ph-base", basePh);
        input.setAttribute("data-os-ph-suffix", needSuffix ? "1" : "0");

        const ph = needSuffix ? `${basePh}${__t("sidebar.tts_field_suffix")}` : basePh;
        input.placeholder = ph;

        input.setAttribute("data-os-persist", `sidebar.tts.${provider}.${key}`);

        input.dataset.ttsKey = key;

        block.appendChild(input);
      }

      this.ttsProviderFieldsHost.appendChild(block);
    }

    try { __osHydratePersistedFields(this.ttsBox || document); } catch {}
    try { __osBindPersistedFields(this.ttsBox || document); } catch {}

    if (before) {
      this.ttsProviderSelect.value = before;
    } else {
      this.ttsProviderSelect.value = "";
    }

    try { this.ttsProviderSelect.dispatchEvent(new Event("change", { bubbles: true })); } catch {}
  }

  // restoreSidebarState() {
  //   const v = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);

  //   if (v === null) {
  //   // 首次访问：默认收起，并写入本地存储（后续刷新保持一致）
  //     document.body.classList.add("sidebar-collapsed");
  //     localStorage.setItem(SIDEBAR_COLLAPSED_KEY, "1");
  //     return;
  //   }

  //   // 已有配置：1 收起，0 展开
  //   document.body.classList.toggle("sidebar-collapsed", v === "1");
  // }

  // restoreDevbarState() {
  //   const v = localStorage.getItem(DEVBAR_COLLAPSED_KEY);

  //   if (v === null) {
  //     // 首次访问：默认收起
  //     document.body.classList.add("devbar-collapsed");
  //     localStorage.setItem(DEVBAR_COLLAPSED_KEY, "1");
  //     return;
  //   }

  //   document.body.classList.toggle("devbar-collapsed", v === "1");
  // }

  _updateSendButtonUI() {
    if (!this.sendBtn) return;

    if (this.streaming) {
      this.sendBtn.innerHTML = this._sendIconStop;
      this.sendBtn.setAttribute("aria-label", "打断");
      this.sendBtn.title = "打断";
    } else {
      this.sendBtn.innerHTML = this._sendIconSend;
      this.sendBtn.setAttribute("aria-label", "发送");
      this.sendBtn.title = "发送";
    }
  }

  async interruptTurn() {
    if (!this.sessionId) return;
    if (!this.streaming) return;
    if (this.canceling) return;

    this.canceling = true;
    this._updateComposerDisabledState();

    try {
      await this.api.cancelTurn(this.sessionId);
      // 不需要本地立刻 finalize，等后端 assistant.end 来收尾并把上下文写干净
    } catch (e) {
      this.canceling = false;
      this._updateComposerDisabledState();
      this.ui.showToastI18n("toast.interrupt_failed", { msg: (e && (e.message || e)) || "" });
      setTimeout(() => this.ui.hideToast(), 1600);
    }
  }


  toggleDevbar() {
    document.body.classList.toggle("devbar-collapsed");
    // const collapsed = document.body.classList.contains("devbar-collapsed");
    // localStorage.setItem(DEVBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
  }

  setDeveloperMode(enabled) {
    const on = !!enabled;
    const devbar = this.devbarEl || $("#devbar");
    if (!devbar) return;

    if (on) {
      document.body.classList.add("dev-mode");
      devbar.classList.remove("hidden");
    } else {
      document.body.classList.remove("dev-mode");
      devbar.classList.add("hidden");
    }
  }

  toggleSidebar() {
    document.body.classList.toggle("sidebar-collapsed");
    // const collapsed = document.body.classList.contains("sidebar-collapsed");
    // localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
  }

  _setLang(lang, { persist = true, syncServer = true } = {}) {
    const v = __osNormLang(lang);
    if (!v) return;

    __applyLang(v, { persist });

    this.lang = v;
    if (persist) this._langWasStored = true;

    if (this.langToggle) this.langToggle.checked = (v === "en");

    this._rerenderLangDynamicBits();

    if (this.ui && typeof this.ui.rerenderToast === "function") {
      this.ui.rerenderToast();
    }

    try { this.ui?.rerenderAssistantPlaceholder?.(); } catch {}
    try { this.ui?.rerenderToolCards?.(); } catch {}
    try { this.ui?.rerenderToolMediaPreviews?.(); } catch {}

    if (syncServer) this._pushLangToServer();
  }

  _rerenderLangDynamicBits() {
    const apply = (sel) => {
      if (!sel) return;
      const opt = sel.querySelector(`option[value="${CUSTOM_MODEL_KEY}"]`);
      if (opt) opt.textContent = __t("sidebar.use_custom_model");
    };

    apply(this.llmSelect);
    apply(this.vlmSelect);

    if (this.ttsProviderSelect) {
      const opt0 = this.ttsProviderSelect.querySelector('option[value=""]');
      if (opt0) opt0.textContent = __t("sidebar.tts_default");
    }

    __rerenderTtsFieldPlaceholders(document);
  }

  _pushLangToServer() {
    if (!this.ws) return;
    this.ws.send("session.set_lang", { lang: this.lang });
  }

  applySnapshotLimits(snapshot) {
    const lim = (snapshot && snapshot.limits) ? snapshot.limits : {};
    const toInt = (v, d) => {
      const n = Number(v);
      return Number.isFinite(n) && n > 0 ? n : d;
    };

    this.limits = {
      max_media_per_session: toInt(lim.max_media_per_session, this.limits.max_media_per_session || 30),
      max_pending_media_per_session: toInt(lim.max_pending_media_per_session, this.limits.max_pending_media_per_session || 30),
      upload_chunk_bytes: toInt(lim.upload_chunk_bytes, this.limits.upload_chunk_bytes || (8 * 1024 * 1024)),
    };
  }

  applySnapshotModels(snapshot) {
    const llmModels =
      (snapshot && Array.isArray(snapshot.llm_models)) ? snapshot.llm_models :
      (snapshot && Array.isArray(snapshot.chat_models)) ? snapshot.chat_models : [];

    const llmCurrent =
      (snapshot && typeof snapshot.llm_model_key === "string") ? snapshot.llm_model_key :
      (snapshot && typeof snapshot.chat_model_key === "string") ? snapshot.chat_model_key : "";

    const vlmModels = (snapshot && Array.isArray(snapshot.vlm_models)) ? snapshot.vlm_models : [];
    const vlmCurrent = (snapshot && typeof snapshot.vlm_model_key === "string") ? snapshot.vlm_model_key : "";

    // 确保至少有一个选项
    const llmList = (llmModels && llmModels.length) ? llmModels.slice() : (llmCurrent ? [llmCurrent] : []);
    const vlmList = (vlmModels && vlmModels.length) ? vlmModels.slice() : (vlmCurrent ? [vlmCurrent] : []);

    this.llmModels = llmList;
    this.vlmModels = vlmList;

    // render LLM select
    if (this.llmSelect) {
      this.llmSelect.innerHTML = "";
      for (const m of llmList) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = (m === CUSTOM_MODEL_KEY) ? __t("sidebar.use_custom_model") : m;
        this.llmSelect.appendChild(opt);
      }
      let selected = "";
      if (llmCurrent && llmList.includes(llmCurrent)) selected = llmCurrent;
      else if (llmList.length) selected = llmList[0];
      this.llmModel = selected || null;
      if (this.llmModel) this.llmSelect.value = this.llmModel;
    }

    // render VLM select
    if (this.vlmSelect) {
      this.vlmSelect.innerHTML = "";
      for (const m of vlmList) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = (m === CUSTOM_MODEL_KEY) ? __t("sidebar.use_custom_model") : m;
        this.vlmSelect.appendChild(opt);
      }
      let selected = "";
      if (vlmCurrent && vlmList.includes(vlmCurrent)) selected = vlmCurrent;
      else if (vlmList.length) selected = vlmList[0];
      this.vlmModel = selected || null;
      if (this.vlmModel) this.vlmSelect.value = this.vlmModel;
    }

    this._syncConfigPanels();
  }


  _syncConfigPanels() {
    const llmCustom = (this.llmModel === CUSTOM_MODEL_KEY);
    const vlmCustom = (this.vlmModel === CUSTOM_MODEL_KEY);

    if (this.customLlmSection) this.customLlmSection.classList.toggle("hidden", !llmCustom);
    if (this.customVlmSection) this.customVlmSection.classList.toggle("hidden", !vlmCustom);

    const provider = (this.ttsProviderSelect && this.ttsProviderSelect.value)
      ? String(this.ttsProviderSelect.value).trim()
      : "";

    const host = this.ttsProviderFieldsHost || $("#ttsProviderFields");
    if (host) {
      host.querySelectorAll("[data-tts-provider]").forEach((el) => {
        const v = String(el.dataset.ttsProvider || "");
        el.classList.toggle("hidden", !provider || v !== provider);
      });
    }

    // ---- Pexels custom key show/hide ----
    const pMode = (this.pexelsKeyModeSelect && this.pexelsKeyModeSelect.value)
      ? String(this.pexelsKeyModeSelect.value).trim()
      : "default";

    const showCustomPexels = (pMode === "custom");
    if (this.pexelsCustomKeyBox) this.pexelsCustomKeyBox.classList.toggle("hidden", !showCustomPexels);
  }


  _readCustomModelsFromUI() {
    const s = (x) => String(x ?? "").trim();
    return {
      llm: {
        model: s(this.customLlmModel?.value),
        base_url: s(this.customLlmBaseUrl?.value),
        api_key: s(this.customLlmApiKey?.value),
      },
      vlm: {
        model: s(this.customVlmModel?.value),
        base_url: s(this.customVlmBaseUrl?.value),
        api_key: s(this.customVlmApiKey?.value),
      },
    };
  }

  _validateCustomModels(cfg, { needLlm = false, needVlm = false } = {}) {
    const llm = cfg?.llm || {};
    const vlm = cfg?.vlm || {};
    const miss = (x) => !x || !String(x).trim();

    if (needLlm && (miss(llm.model) || miss(llm.base_url) || miss(llm.api_key))) {
      return "custom llm config is incomplete: please fill in model/base_url/api_key";
    }
    if (needVlm && (miss(vlm.model) || miss(vlm.base_url) || miss(vlm.api_key))) {
      return "custom vlm config is incomplete: please fill in model/base_url/api_key";
    }
    return "";
  }


  _readTtsConfigFromUI() {
    const provider = (this.ttsProviderSelect && this.ttsProviderSelect.value)
      ? String(this.ttsProviderSelect.value).trim()
      : "";
    if (!provider) return null;

    const host = this.ttsProviderFieldsHost || $("#ttsProviderFields");
    const params = {};

    if (host) {
      const block = host.querySelector(`[data-tts-provider="${provider}"]`);
      if (block) {
        const fields = block.querySelectorAll("input[data-tts-key], select[data-tts-key], textarea[data-tts-key]");
        fields.forEach((el) => {
          const k = String(el.dataset.ttsKey || "").trim();
          if (!k) return;
          const v = String(el.value ?? "").trim();
          if (v !== "") params[k] = v; 
        });
      }
    }

    // 统一 payload：{ provider, <provider>:{...} }
    const out = { provider };
    out[provider] = params; // 允许为空 {}
    return out;
  }

  _readPexelsConfigFromUI() {
    if (!this.pexelsKeyModeSelect) return null;

    const modeRaw = String(this.pexelsKeyModeSelect.value || "").trim();
    const mode = (modeRaw === "custom") ? "custom" : "default";

    let api_key = "";
    if (mode === "custom" && this.pexelsApiKeyInput) {
      api_key = String(this.pexelsApiKeyInput.value || "").trim();
    }

    return { mode, api_key };
  }

  // ----- Session history (localStorage, per-browser) -----

  _loadSessionHistory() {
    try {
      const raw = localStorage.getItem(SESSION_LIST_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];

      const out = [];
      for (const item of parsed) {
        if (!item || typeof item !== "object") continue;
        const id = String(item.id || "").trim();
        if (!id) continue;
        const title = String(item.title || "").trim();
        const created_at = Number(item.created_at || Date.now());
        const updated_at = Number(item.updated_at || created_at);
        const last_preview = String(item.last_preview || "").trim();
        const has_user = !!item.has_user;
        out.push({ id, title, created_at, updated_at, last_preview, has_user });
      }
      return out;
    } catch {
      return [];
    }
  }

  _saveSessionHistory(list) {
    if (!Array.isArray(list)) return;
    try {
      // 限制列表长度，避免无限增长；不改变原有顺序
      const trimmed = list.slice(0, 50);
      localStorage.setItem(SESSION_LIST_KEY, JSON.stringify(trimmed));
    } catch {
      // 忽略存储错误（Safari 无痕等）
    }
  }

  _newChatTitle() {
    const lang = __osNormLang(this.lang || "zh");
    return (lang === "en") ? "New chat" : "新对话";
  }

  _isNewChatTitle(title) {
    const t = String(title || "").trim();
    if (!t) return true;
    return t === "新对话" || t === "New chat";
  }

  _deriveSessionTitle(snapshot, fallbackId) {
    const history = (snapshot && Array.isArray(snapshot.history)) ? snapshot.history : [];
    const firstUser = history.find((h) => h && h.role === "user" && typeof h.content === "string");
    let title = firstUser ? String(firstUser.content || "") : "";
    title = title.replace(/\s+/g, " ").trim();
    if (title) {
      if (title.length > 32) title = title.slice(0, 32) + "…";
      return title;
    }
    // 还没输入任何内容：显示“新对话”
    return this._newChatTitle();
  }

  _deriveSessionPreview(snapshot) {
    const history = (snapshot && Array.isArray(snapshot.history)) ? snapshot.history : [];
    // 优先最后一条用户消息，其次助手
    let last = null;
    for (let i = history.length - 1; i >= 0; i--) {
      const it = history[i];
      if (!it || typeof it !== "object") continue;
      if (!last && (it.role === "assistant" || it.role === "user") && typeof it.content === "string") {
        last = it;
      }
      if (it.role === "user" && typeof it.content === "string") {
        last = it;
        break;
      }
    }
    if (!last || typeof last.content !== "string") return "";
    let text = String(last.content || "").replace(/\s+/g, " ").trim();
    if (!text) return "";
    if (text.length > 40) text = text.slice(0, 40) + "…";
    return text;
  }

  _snapshotHasUserMessage(snapshot) {
    const history = (snapshot && Array.isArray(snapshot.history)) ? snapshot.history : [];
    return history.some((h) => h && h.role === "user" && typeof h.content === "string" && String(h.content || "").trim());
  }

  _upsertSessionHistoryFromSnapshot(sessionId, snapshot) {
    const sid = String(sessionId || "").trim();
    if (!sid) return;
    if (!Array.isArray(this.sessionHistory)) this.sessionHistory = [];

    const now = Date.now();
    const title = this._deriveSessionTitle(snapshot, sid);
    const preview = this._deriveSessionPreview(snapshot);
    const hasUser = this._snapshotHasUserMessage(snapshot);

    let found = false;
    const next = this.sessionHistory.map((item) => {
      if (item.id !== sid) return item;
      found = true;
      return {
        ...item,
        title: title || item.title || "",
        last_preview: preview || item.last_preview || "",
        // updated_at 仅作元信息保存，不参与排序；更新不会影响列表顺序
        updated_at: now,
        has_user: (item.has_user || false) || hasUser,
      };
    });

    if (!found) {
      next.push({
        id: sid,
        title,
        created_at: now,
        updated_at: now,
        last_preview: preview,
        has_user: hasUser,
      });
    }

    this.sessionHistory = next;
    this._saveSessionHistory(this.sessionHistory);
  }

  _removeSessionFromHistory(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid || !Array.isArray(this.sessionHistory)) return;
    const next = this.sessionHistory.filter((item) => item.id !== sid);
    if (next.length === this.sessionHistory.length) return;
    this.sessionHistory = next;
    this._saveSessionHistory(this.sessionHistory);
  }

  _renderSessionHistory(activeId) {
    const host = this.sessionHistoryListEl || $("#sessionHistoryList");
    if (!host) return;

    host.innerHTML = "";

    const list = Array.isArray(this.sessionHistory) ? this.sessionHistory.slice() : [];
    if (!list.length) {
      const empty = document.createElement("div");
      empty.className = "session-history-empty";
      empty.textContent = __t("sidebar.history_empty");
      host.appendChild(empty);
      return;
    }

    // 始终按创建时间倒序展示，点击切换会话不会改变顺序
    list.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

    for (const item of list) {
      if (!item || !item.id) continue;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "session-history-item";
      btn.dataset.sessionId = String(item.id);
      if (activeId && String(activeId) === String(item.id)) {
        btn.classList.add("is-active");
      }

      const titleEl = document.createElement("div");
      titleEl.className = "session-history-title";
      titleEl.textContent = item.title || this._newChatTitle();
      btn.appendChild(titleEl);

      if (item.last_preview) {
        const metaEl = document.createElement("div");
        metaEl.className = "session-history-meta";
        metaEl.textContent = item.last_preview;
        btn.appendChild(metaEl);
      }

      host.appendChild(btn);
    }
  }

  _getSessionHistoryItem(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid || !Array.isArray(this.sessionHistory)) return null;
    return this.sessionHistory.find((x) => x && x.id === sid) || null;
  }

  _isCurrentSessionBlank() {
    if (!this.sessionId) return false;
    const item = this._getSessionHistoryItem(this.sessionId);
    const hasUser = !!(item && item.has_user);
    // “新对话/草稿会话”的判定：只看是否出现过用户消息。
    // 说明：上传了素材但还没发送第一条消息时，仍然应该被视为同一个“新对话”，避免无限创建多个新会话。
    return !hasUser;
  }

  _findAnyBlankSessionId() {
    if (!Array.isArray(this.sessionHistory) || !this.sessionHistory.length) return null;
    // 复用“最新的空会话”，保证“全局只保留一个新对话”的交互更符合直觉
    const blanks = this.sessionHistory
      .filter((it) => it && !it.has_user && it.id)
      .slice()
      .sort((a, b) => (Number(b.created_at) || 0) - (Number(a.created_at) || 0));
    return blanks.length ? String(blanks[0].id || "") : null;
  }

  _touchBlankSessionAsNewChat(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid) return;
    const now = Date.now();
    const next = (Array.isArray(this.sessionHistory) ? this.sessionHistory : []).map((it) => {
      if (!it || it.id !== sid) return it;
      return {
        ...it,
        title: this._newChatTitle(),
        last_preview: "",
        has_user: false,
        created_at: now,  // 视为“新对话”，刷新创建时间使其成为最新
        updated_at: now,
      };
    });
    this.sessionHistory = next;
    this._saveSessionHistory(this.sessionHistory);
    this._renderSessionHistory(this.sessionId);
  }

  _handleMissingSessionOnClick(sessionId) {
    // 遇到错误时，直接从本地列表移除，并用 toast 提示
    const sid = String(sessionId || "").trim();
    if (!sid) return;

    this._removeSessionFromHistory(sid);
    this._renderSessionHistory(this.sessionId);

    const lang = __osNormLang(this.lang || "zh");
    const msg = (lang === "en")
      ? "This chat has expired or is no longer available and has been removed from history."
      : "该会话已失效或不可用，已从历史列表中移除。";
    if (this.ui && typeof this.ui.showToast === "function") {
      this.ui.showToast(msg);
      setTimeout(() => this.ui.hideToast(), 2000);
    }
  }

  _markSessionHasUserNow(userText) {
    const sid = String(this.sessionId || "").trim();
    if (!sid) return;

    const text = String(userText || "").replace(/\s+/g, " ").trim();
    const preview = text ? (text.length > 40 ? text.slice(0, 40) + "…" : text) : "";
    const now = Date.now();

    if (!Array.isArray(this.sessionHistory)) this.sessionHistory = [];

    let found = false;
    const next = this.sessionHistory.map((it) => {
      if (!it || it.id !== sid) return it;
      found = true;
      const nextTitle = (this._isNewChatTitle(it.title) || !String(it.title || "").trim())
        ? (text.length > 32 ? text.slice(0, 32) + "…" : (text || this._newChatTitle()))
        : it.title;

      return {
        ...it,
        title: nextTitle,
        last_preview: preview || it.last_preview || "",
        has_user: true,
        updated_at: now,
      };
    });

    if (!found) {
      next.push({
        id: sid,
        title: text ? (text.length > 32 ? text.slice(0, 32) + "…" : text) : this._newChatTitle(),
        created_at: now,
        updated_at: now,
        last_preview: preview,
        has_user: true,
      });
    }

    this.sessionHistory = next;
    this._saveSessionHistory(this.sessionHistory);
    this._renderSessionHistory(this.sessionId);
  }

  _flashSessionHistoryItem(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid) return;
    const host = this.sessionHistoryListEl || $("#sessionHistoryList");
    if (!host) return;

    const el = host.querySelector(`[data-session-id="${sid}"]`);
    if (!el) return;

    el.classList.add("flash");
    setTimeout(() => {
      el.classList.remove("flash");
    }, 160);
  }


  _makeChatSendPayload(text, attachment_ids) {
    const payload = { text, attachment_ids, lang: this.lang || "zh" };

    if (this.llmModel) payload.llm_model = this.llmModel;
    if (this.vlmModel) payload.vlm_model = this.vlmModel;

    const rc = {};

    const needLlmCustom = (this.llmModel === CUSTOM_MODEL_KEY);
    const needVlmCustom = (this.vlmModel === CUSTOM_MODEL_KEY);

    if (needLlmCustom || needVlmCustom) {
      const cm = this._readCustomModelsFromUI();
      const err = this._validateCustomModels(cm, { needLlm: needLlmCustom, needVlm: needVlmCustom });
      if (err) return { error: err };

      rc.custom_models = {};
      if (needLlmCustom) rc.custom_models.llm = cm.llm;
      if (needVlmCustom) rc.custom_models.vlm = cm.vlm;
    }

    const tts = this._readTtsConfigFromUI();
    if (tts) rc.tts = tts;

    const pexels = this._readPexelsConfigFromUI();
    if (pexels) {
      rc.search_media = { pexels };
    }

    if (Object.keys(rc).length) payload.service_config = rc;
    return { payload };
  }


  setChatModel(model) {
    const m = String(model || "").trim();
    if (!m) return;
    this.chatModel = m;
  }


  clearLocalObjectUrls() {
    for (const [, url] of this.localObjectUrlByMediaId) {
      try { URL.revokeObjectURL(url); } catch {}
    }
    this.localObjectUrlByMediaId.clear();
  }

  bindLocalUrlsToMedia(list) {
    const arr = Array.isArray(list) ? list : [];
    return arr.map((a) => {
      const url = a && a.id ? this.localObjectUrlByMediaId.get(a.id) : null;
      return url ? { ...a, local_url: url } : a;
    });
  }

  revokeLocalUrl(mediaId) {
    const url = this.localObjectUrlByMediaId.get(mediaId);
    if (url) {
      try { URL.revokeObjectURL(url); } catch {}
      this.localObjectUrlByMediaId.delete(mediaId);
    }
  }

  _updateComposerDisabledState() {
    // - streaming=true：sendBtn 是“打断键”，必须可点（除非正在 canceling）
    // - streaming=false：uploading=true 时不能发送 => 禁用
    const disableSend = this.canceling ? true : (!this.streaming && this.uploading);
    if (this.sendBtn) this.sendBtn.disabled = disableSend;

    if (this.uploadBtn) this.uploadBtn.disabled = !!this.uploading;

    this._updateSendButtonUI();
  }

  _autosizePrompt() {
    const el = this.promptInput;
    if (!el) return;

    // 读取 CSS 的 max-height（比如 180px），读不到就 fallback
    const cs = window.getComputedStyle(el);
    const mh = parseFloat(cs.maxHeight);
    const maxPx = Number.isFinite(mh) && mh > 0 ? mh : 180;

    // 先让它回到 auto，才能正确拿到 scrollHeight
    el.style.height = "auto";

    const next = Math.min(el.scrollHeight, maxPx);
    el.style.height = next + "px";

    // 没超过上限：隐藏滚动条；超过上限：出现滚动条
    el.style.overflowY = (el.scrollHeight > maxPx) ? "auto" : "hidden";
  }

  _nextQuickPromptText() {
    const list = Array.isArray(QUICK_PROMPTS) ? QUICK_PROMPTS : [];
    if (!list.length) return "";

    const idx = (Number(this._quickPromptIdx) || 0) % list.length;
    this._quickPromptIdx = idx + 1;

    const item = list[idx];
    const lang = __osNormLang(this.lang || "zh");

    if (typeof item === "string") return item.trim();
    if (item && typeof item === "object") {
      const v = item[lang] ?? item.zh ?? item.en ?? "";
      return String(v || "").trim();
    }
    return String(item ?? "").trim();
  }

  _insertIntoPrompt(text) {
    const el = this.promptInput;
    const insertText = String(text || "").trim();
    if (!el || !insertText) return;

    const cur = String(el.value || "");

    if (!cur.trim()) {
      el.value = insertText;
      try { el.setSelectionRange(el.value.length, el.value.length); } catch {}
      el.focus();
      this._autosizePrompt();
      return;
    }

    const start = (typeof el.selectionStart === "number") ? el.selectionStart : cur.length;
    const end = (typeof el.selectionEnd === "number") ? el.selectionEnd : cur.length;

    const before = cur.slice(0, start);
    const after = cur.slice(end);

    const isCollapsed = start === end;
    const isAtEnd = isCollapsed && end === cur.length;

    const sep = (isAtEnd && before && !before.endsWith("\n")) ? "\n" : "";

    el.value = before + sep + insertText + after;

    const caret = (before + sep + insertText).length;
    try { el.setSelectionRange(caret, caret); } catch {}

    el.focus();
    this._autosizePrompt();
  }

  bindUI() {
    // sidebar
    if (this.sidebarToggleBtn) {
      this.sidebarToggleBtn.addEventListener("click", () => this.toggleSidebar());
    }
    if (this.createDialogBtn) {
      this.createDialogBtn.addEventListener("click", () => {
        if (this._isSwitchBlocked()) {
          this._showSwitchWhileStreamingToast();
          return;
        }
        this.newSession();
      });
    }

    if (this.sessionHistoryListEl && !this._sessionHistoryBound) {
      this._sessionHistoryBound = true;
      this.sessionHistoryListEl.addEventListener("click", async (e) => {
        const target = e.target.closest("[data-session-id]");
        if (!target) return;

        // 正在生成回复时禁止切换会话，避免未完成内容丢失
        if (this._isSwitchBlocked()) {
          this._showSwitchWhileStreamingToast();
          return;
        }

        const sid = String(target.dataset.sessionId || "").trim();
        if (!sid || sid === this.sessionId) return;

        try {
          const snap = await this.api.getSession(sid);
          await this.useSession(sid, snap);
        } catch (err) {
          console.warn("[session] failed to restore session", sid, err);
          // 仅当明确 404（会话不存在）时才清理本地记录；其它错误不要误删
          if (err && err.status === 404) {
            this._handleMissingSessionOnClick(sid);
            return;
          }

          const lang = __osNormLang(this.lang || "zh");
          const msg = (lang === "en")
            ? "Failed to load this chat for now. Please try again later."
            : "暂时无法加载该会话，请稍后重试。";
          if (this.ui && typeof this.ui.showToast === "function") {
            this.ui.showToast(msg);
            setTimeout(() => this.ui.hideToast(), 2000);
          }
        }
      });
    }

    if (this.llmSelect) {
      this.llmSelect.addEventListener("change", () => {
        const v = (this.llmSelect.value || "").trim();
        if (v) this.llmModel = v;
        this._syncConfigPanels();
      });
    }

    if (this.vlmSelect) {
      this.vlmSelect.addEventListener("change", () => {
        const v = (this.vlmSelect.value || "").trim();
        if (v) this.vlmModel = v;
        this._syncConfigPanels();
      });
    }

    if (this.ttsProviderSelect) {
      this.ttsProviderSelect.addEventListener("change", () => this._syncConfigPanels());
    }

    if (this.pexelsKeyModeSelect) {
      this.pexelsKeyModeSelect.addEventListener("change", () => this._syncConfigPanels());
    }

    // devbar toggle（仅 developer_mode=true 时 devbar 会显示）
    if (this.devbarToggleBtn) {
      this.devbarToggleBtn.addEventListener("click", () => this.toggleDevbar());
    }

    // uploader
    this.uploadBtn.addEventListener("click", () => this.fileInput.click());

    this.fileInput.addEventListener("change", async () => {
      let files = Array.from(this.fileInput.files || []);
      this.fileInput.value = "";
      if (!files.length) return;

      // 会话内 pending 上限
      const maxPending = Number(this.limits.max_pending_media_per_session || 30);
      const remain = Math.max(0, maxPending - (this.pendingMedia.length || 0));
      if (remain <= 0) {
        this.ui.showToastI18n("toast.pending_limit", { max: maxPending });
        setTimeout(() => this.ui.hideToast(), 1600);
        return;
      }
      if (files.length > remain) {
        this.ui.showToastI18n("toast.pending_limit_partial", { remain, max: maxPending });
        setTimeout(() => this.ui.hideToast(), 1400);
        files = files.slice(0, remain);
      }

      const totalBytes = Math.max(1, files.reduce((s, f) => s + (f.size || 0), 0));
      let confirmedBytesAll = 0;

      this.uploading = true;
      this._updateComposerDisabledState();

      try {
        this.ui.showToastI18n("toast.uploading", { pct: 0 });

        // 分片
        for (let i = 0; i < files.length; i++) {
          const f = files[i];

          // 预先创建 ObjectURL（用于 (3) 预览走本地缓存）
          const localUrl = URL.createObjectURL(f);

          try {
            const resp = await this.api.uploadMediaChunked(this.sessionId, f, {
              chunkSize: this.limits.upload_chunk_bytes,
              onProgress: (loadedInFile, fileTotal) => {
                const overallLoaded = Math.min(totalBytes, confirmedBytesAll + (loadedInFile || 0));
                const pct = Math.round((overallLoaded / totalBytes) * 100);
                this.ui.showToastI18n("toast.uploading_file", { i: i + 1, n: files.length, name: f.name, pct });
              },
            });

            // 上传完成：把 media_id -> localUrl 绑定起来
            if (resp && resp.media && resp.media.id) {
              this.localObjectUrlByMediaId.set(resp.media.id, localUrl);
            } else {
              // 理论不应发生；发生就释放
              try { URL.revokeObjectURL(localUrl); } catch {}
            }

            confirmedBytesAll += (f.size || 0);

            // pending 更新（绑定 local_url 后再渲染）
            this.setPending((resp && resp.pending_media) ? resp.pending_media : []);
          } catch (e) {
            // 本文件失败：释放 URL，避免泄漏
            try { URL.revokeObjectURL(localUrl); } catch {}
            throw e;
          }
        }

        this.ui.hideToast();
      } catch (e) {
        this.ui.hideToast();
        this.ui.showToastI18n("toast.upload_failed", { msg: (e && (e.message || e)) || "" });
        setTimeout(() => this.ui.hideToast(), 1800);
      } finally {
        this.uploading = false;
        this._updateComposerDisabledState();
      }
    });


    // pending 删除：用事件委托
    $("#pendingRow").addEventListener("click", async (e) => {
      const el = e.target;
      if (!el.classList.contains("media-remove")) return;
      const mediaId = el.dataset.mediaId;
      if (!mediaId) return;

      try {
        const resp = await this.api.deletePendingMedia(this.sessionId, mediaId);
        this.revokeLocalUrl(mediaId);
        this.setPending(resp.pending_media || []);
      } catch (err) {
        this.ui.showToastI18n("toast.delete_failed", { msg: (err && (err.message || err)) || "" });
        setTimeout(() => this.ui.hideToast(), 1600);
      }
    });

    // send
    this.sendBtn.addEventListener("click", () => this.sendPrompt({ source: "button" }));
    this.promptInput.addEventListener("keydown", (e) => {
      // 避免中文输入法“正在组词/选词”时按 Enter 误触发发送
      if (e.isComposing || e.keyCode === 229) return;

      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendPrompt({ source: "enter" });
      }
    });

    //quick prompt fill
    if (this.quickPromptBtn && !this._quickPromptBound) {
      this._quickPromptBound = true;

      this.quickPromptBtn.addEventListener("click", (e) => {
        e.preventDefault();

        const t = this._nextQuickPromptText();
        if (!t) return;

        this.promptInput.value = t;
        this._autosizePrompt();
        this.promptInput.focus();
        try { this.promptInput.setSelectionRange(t.length, t.length); } catch {}

        this.quickPromptBtn.classList.add("is-active");
        setTimeout(() => this.quickPromptBtn.classList.remove("is-active"), 160);
      });
    }

    // PATCH: prompt 自动长高
    if (this.promptInput && !this._promptAutoResizeBound) {
      this._promptAutoResizeBound = true;

      const resize = () => this._autosizePrompt();
      this.promptInput.addEventListener("input", resize);
      window.addEventListener("resize", resize, { passive: true });

      // 首次初始化/切换会话后确保高度正确
      requestAnimationFrame(resize);
    }

    // lang toggle
    if (this.langToggle) {
      this.langToggle.checked = (this.lang === "en");

      this.langToggle.addEventListener("change", () => {
        const next = this.langToggle.checked ? "en" : "zh";
        this._setLang(next, { persist: true, syncServer: true });
      });
    }
  }

  setPending(list) {
    const arr = this.bindLocalUrlsToMedia(Array.isArray(list) ? list : []);
    this.pendingMedia = arr;
    this.ui.renderPendingMedia(this.pendingMedia);
  }

  async newSession() {
    // 如果当前会话仍为空（未输入任何内容），再次点击“创建新对话”不新建 session：
    // - 保持在当前界面
    // - 仅刷新时间（让它被视为最新的新对话）
    if (this._isCurrentSessionBlank()) {
      this._touchBlankSessionAsNewChat(this.sessionId);
      this._flashSessionHistoryItem(this.sessionId);
      return;
    }

    // 当前不是“新对话”：如果历史中已经存在一个空会话，则直接切换过去，而不是创建第二个“新对话”
    const blankId = this._findAnyBlankSessionId();
    if (blankId && blankId !== this.sessionId) {
      try {
        const snap = await this.api.getSession(blankId);
        await this.useSession(blankId, snap);
        return;
      } catch (e) {
        console.warn("[session] failed to reuse blank session, will create new one:", e);
        // 仅当明确 404 时才清理本地记录；其它错误不要误删
        if (e && e.status === 404) {
          this._removeSessionFromHistory(blankId);
          this._renderSessionHistory(this.sessionId);
        }
      }
    }

    const snap = await this.api.createSession();
    await this.useSession(snap.session_id, snap);
  }

  async useSession(sessionId, snapshot) {
    this.streaming = false;
    this.uploading = false;
    this._updateComposerDisabledState();

    this.sessionId = sessionId;

    const snapLang = snapshot && snapshot.lang;
    if (!this._langWasStored && snapLang) {
      this._setLang(snapLang, { persist: true, syncServer: false });
    } else {
      this._setLang(this.lang, { persist: false, syncServer: false });
    }

    // 切会话：清掉上一会话的本地缓存 URL，避免泄漏
    this.clearLocalObjectUrls();

    // 从后端 snapshot 读取 limits（按素材个数限制/分片大小等）
    this.applySnapshotLimits(snapshot);
    this.applySnapshotModels(snapshot);

    localStorage.setItem(SESSION_ID_KEY, sessionId);

    // 会话列表：更新/插入并重新渲染
    this._upsertSessionHistoryFromSnapshot(sessionId, snapshot);
    this._renderSessionHistory(sessionId);

    this.setDeveloperMode(!!snapshot.developer_mode);

    this.ui.setSessionId(sessionId);
    this.ui.clearAll();

    // 回放 history
    const history = snapshot.history || [];
    for (const item of history) {
      if (item.role === "user") {
        this.ui.appendUserMessage(item.content || "", item.attachments || []);
      } else if (item.role === "assistant") {
        this.ui.startAssistantMessage({placeholder: false});
        this.ui.finalizeAssistant(item.content || "");
      } else if (item.role === "tool") {
        this.ui.upsertToolCard(item.tool_call_id, {
          server: item.server,
          name: item.name,
          state: item.state,
          args: item.args,
          progress: item.progress,
          message: item.message,
          summary: item.summary,
        });

        if (item.summary != null) {
          this.ui.appendDevSummary(item.tool_call_id, {
            server: item.server,
            name: item.name,
            summary: item.summary,
            is_error: item.state === "error",
          });
        }
      }
    }

    this.setPending(snapshot.pending_media || []);
    this.connectWs();

    // 切换到会话时，让对应历史项轻微闪烁，增强反馈
    this._flashSessionHistoryItem(sessionId);
  }

  connectWs() {
    if (this.ws) this.ws.close();

    this.ws = new WsClient(this.wsUrl(this.sessionId), (evt) => this.onWsEvent(evt));
    this.ws.connect();
  }

  onWsEvent(evt) {
    const { type, data } = evt || {};
    if (type === "session.snapshot") {
      // 一般用不上（useSession 已经回放了），但保留兼容
      this.setDeveloperMode(!!data.developer_mode);
      this.ui.setSessionId(data.session_id);
      this.applySnapshotModels(data || {});

      const serverLang = data && data.lang;
      const sv = __osNormLang(serverLang);
      if (sv && sv !== this.lang) {
        if (this._langWasStored) {
          this._pushLangToServer();
        } else {
          this._setLang(sv, { persist: true, syncServer: false });
        }
      }

      this.setPending(data.pending_media || []);
      return;
    }

    if (type === "chat.user") {
      // 以服务端为准更新 pending（避免客户端/服务端状态漂移）
      this.setPending((data || {}).pending_media || []);
      return;
    }

    if (type === "assistant.start") {
      // 请求已开始流式：解除“发送后短暂锁”
      this._clearSwitchLock();
      this.streaming = true;
      this._updateComposerDisabledState();
      // 如果之前有 placeholder（rawText 为空），先清掉避免残留
      try { this.ui.flushAssistantSegment(); } catch {}
      this.ui.startAssistantMessage({placeholder: true});
      return;
    }

    if (type === "assistant.flush") {
      this.ui.flushAssistantSegment();
      return;
    }

    if (type === "assistant.delta") {
      this.ui.appendAssistantDelta((data || {}).delta || "");
      return;
    }

    if (type === "assistant.end") {
      this.streaming = false;
      this.canceling = false;
      this._updateComposerDisabledState();
      this.ui.endAssistantTurn((data || {}).text || "");
      return;
    }

    if (type === "tool.start") {
      this.ui.upsertToolCard(data.tool_call_id, {
        server: data.server,
        name: data.name,
        state: "running",
        args: data.args || {},
        progress: 0,
      });
      return;
    }

    if (type === "tool.progress") {
      this.ui.upsertToolCard(data.tool_call_id, {
        server: data.server,
        name: data.name,
        state: "running",
        progress: typeof data.progress === "number" ? data.progress : 0,
        message: data.message || "",
        __progress_mode: "real",
      });
      return;
    }

    if (type === "tool.end") {
      this.ui.upsertToolCard(data.tool_call_id, {
        server: data.server,
        name: data.name,
        state: data.is_error ? "error" : "success",
        summary: (data && Object.prototype.hasOwnProperty.call(data, "summary")) ? data.summary : null,
      });
      this.ui.appendDevSummary(data.tool_call_id, {
        server: data.server,
        name: data.name,
        summary: data.summary,
        is_error: !!data.is_error,
      });
      return;
    }

    if (type === "chat.cleared") {
      this._clearSwitchLock();
      this.streaming = false;
      this.canceling = false;
      this._updateComposerDisabledState();
      this.ui.clearAll();
      return;
    }

    if (type === "error") {
      this._clearSwitchLock();
      this.streaming = false;
      this.canceling = false;
      this._updateComposerDisabledState();

      const msg = String((data || {}).message || "unknown error");
      const partial = String((data || {}).partial_text || "").trim();

      // 用 endAssistantTurn 结束当前流式气泡：
      // - 有 partial：保留已输出内容，并追加错误说明
      // - 无 partial：直接显示错误
      const text = partial
        ? `${partial}\n\n（发生错误：${msg}）`
        : `发生错误：${msg}`;

      this.ui.endAssistantTurn(text);
      return;
    }
  }

  sendPrompt({ source = "button" } = {}) {
    if (!this.ws) return;

    const text = (this.promptInput.value || "").trim();

    if (this.streaming) {
      // Enter 防误触：输入为空 -> 不打断、不发送
      if (source === "enter" && !text) {
        return;
      }

      // Enter 且有文本：打断 + 发送新 prompt
      if (source === "enter" && text) {
        if (this.canceling) return;

        // 上传中提示并仅打断（让旧输出停掉），等用户上传完再回车发送
        if (this.uploading) {
          this.ui.showToastI18n("toast.uploading_interrupt_send", {});
          setTimeout(() => this.ui.hideToast(), 1600);
          this.interruptTurn(); // 有意图（非空）=> 仍然打断
          return;
        }

        const attachments = this.pendingMedia.slice();
        const attachment_ids = attachments.map(a => a.id);

        // 1) 立即在 UI 插入 user 气泡（体验更顺滑）
        this.ui.appendUserMessage(text, attachments);
        this.setPending([]);

        // 本地先把会话标记为“已输入”，避免历史里一直显示“新对话”
        this._markSessionHasUserNow(text);

        // 2) 清空输入框
        this.promptInput.value = "";
        this._autosizePrompt();

        // 3) 触发打断（异步，不 await）
        this.interruptTurn();

        // 4) 立即把新消息发到 WS（服务器会在旧 turn 结束后按序处理）
        const built = this._makeChatSendPayload(text, attachment_ids);
        if (built.error) {
          this.ui.showToast(built.error);
          setTimeout(() => this.ui.hideToast(), 1800);
          return;
        }
        // 发送后到 assistant.start 之间加锁，禁止切会话
        this._setSwitchLock();
        this.ws.send("chat.send", built.payload);

        return;
      }

      // 其它情况（按钮点击/停止图标）：打断
      this.interruptTurn();
      return;
    }

    // -----------------------------
    // 非 streaming：正常发送
    // -----------------------------
    if (this.uploading) {
      this.ui.showToastI18n("toast.uploading_cannot_send", {});
      setTimeout(() => this.ui.hideToast(), 1400);
      return;
    }

    if (!text) return;

    const attachments = this.pendingMedia.slice();
    const attachment_ids = attachments.map(a => a.id);

    this.ui.appendUserMessage(text, attachments);
    this.setPending([]);

    // 本地先把会话标记为“已输入”，避免历史里一直显示“新对话”
    this._markSessionHasUserNow(text);

    this.promptInput.value = "";
    this._autosizePrompt();

    const built = this._makeChatSendPayload(text, attachment_ids);
    if (built.error) {
      this.ui.showToast(built.error);
      setTimeout(() => this.ui.hideToast(), 1800);
      return;
    }
    // 发送后到 assistant.start 之间加锁，禁止切会话（不插入任何硬编码占位气泡）
    this._setSwitchLock();
    this.ws.send("chat.send", built.payload);
  }

}

new App().bootstrap();
/* =========================================================
   PATCH (mobile viewport / keyboard safe area)
   - updates CSS vars: --kb, --composer-h, --vvh
   ========================================================= */
(function () {
  const root = document.documentElement;
  const composer = document.querySelector(".composer");
  if (!root || !composer) return;

  let raf = 0;

  const compute = () => {
    raf = 0;

    const vv = window.visualViewport;
    const layoutH = window.innerHeight || document.documentElement.clientHeight || 0;

    const vvH = vv ? vv.height : layoutH;
    const vvTop = vv ? vv.offsetTop : 0;

    // Keyboard overlay height (0 on most desktops)
    const kb = vv ? Math.max(0, layoutH - (vvH + vvTop)) : 0;

    root.style.setProperty("--vvh", `${Math.round(vvH)}px`);
    root.style.setProperty("--kb", `${Math.round(kb)}px`);

    const ch = composer.getBoundingClientRect().height || 0;
    if (ch > 0) root.style.setProperty("--composer-h", `${Math.round(ch)}px`);
  };

  const schedule = () => {
    if (raf) return;
    raf = requestAnimationFrame(compute);
  };

  compute();

  // Window resize / orientation
  window.addEventListener("resize", schedule, { passive: true });
  window.addEventListener("orientationchange", schedule, { passive: true });

  // iOS: focusing inputs changes visual viewport
  document.addEventListener("focusin", schedule, true);
  document.addEventListener("focusout", schedule, true);

  // visualViewport gives the best signal on mobile browsers
  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", schedule, { passive: true });
    window.visualViewport.addEventListener("scroll", schedule, { passive: true });
  }

  // composer height changes (pending bar / textarea autosize)
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(schedule);
    ro.observe(composer);
  }
})();

/* =========================================================
   Persist sidebar config across refresh (keys, base_url, etc.)
   ========================================================= */

const __OS_PERSIST_STORAGE = window.sessionStorage; // <- 改成 localStorage 即可“关浏览器也还在”
const __OS_PERSIST_KEY = "openstoryline_user_config_v1";

function __osSafeParseJson(s, fallback) {
  try {
    const v = JSON.parse(s);
    return (v && typeof v === "object") ? v : fallback;
  } catch {
    return fallback;
  }
}

function __osLoadConfig() {
  return __osSafeParseJson(__OS_PERSIST_STORAGE.getItem(__OS_PERSIST_KEY), {});
}

function __osSaveConfig(cfg) {
  try {
    __OS_PERSIST_STORAGE.setItem(__OS_PERSIST_KEY, JSON.stringify(cfg || {}));
  } catch (e) {
    console.warn("[persist] save failed:", e);
  }
}

function __osGetByPath(obj, path) {
  if (!obj || !path) return undefined;
  const parts = String(path).split(".").filter(Boolean);
  let cur = obj;
  for (const p of parts) {
    if (!cur || typeof cur !== "object") return undefined;
    cur = cur[p];
  }
  return cur;
}

function __osSetByPath(obj, path, value) {
  const parts = String(path).split(".").filter(Boolean);
  if (!parts.length) return;
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i];
    if (!cur[k] || typeof cur[k] !== "object") cur[k] = {};
    cur = cur[k];
  }
  cur[parts[parts.length - 1]] = value;
}

const __osPendingSelectValues = new Map();

function __osApplySelectValue(selectEl, desiredValue) {
  const desired = String(desiredValue ?? "");
  const before = selectEl.value;
  selectEl.value = desired;

  const ok = selectEl.value === desired;
  if (ok && before !== selectEl.value) {
    // 触发你现有的 UI 联动逻辑（显示/隐藏 box 等）
    selectEl.dispatchEvent(new Event("change", { bubbles: true }));
  }
  return ok;
}

function __osObserveSelectOptions(selectEl) {
  if (selectEl.__osSelectObserver) return;

  const observer = new MutationObserver(() => {
    const desired = __osPendingSelectValues.get(selectEl);
    if (desired == null) return;

    if (__osApplySelectValue(selectEl, desired)) {
      __osPendingSelectValues.delete(selectEl);
      observer.disconnect();
      selectEl.__osSelectObserver = null;
    }
  });

  observer.observe(selectEl, { childList: true, subtree: true });
  selectEl.__osSelectObserver = observer;
}

function __osHydratePersistedFields(root = document) {
  const cfg = __osLoadConfig();
  const nodes = root.querySelectorAll("[data-os-persist]");

  nodes.forEach((el) => {
    const key = el.getAttribute("data-os-persist");
    if (!key) return;

    const v = __osGetByPath(cfg, key);
    if (v == null) return;

    const tag = (el.tagName || "").toLowerCase();
    const type = String(el.type || "").toLowerCase();

    try {
      if (type === "checkbox") {
        el.checked = !!v;
      } else if (tag === "select") {
        // 如果选项是异步加载的（比如 modelSelect），先尝试设置，不行就等 options 出来再设置
        if (!__osApplySelectValue(el, v)) {
          __osPendingSelectValues.set(el, String(v));
          __osObserveSelectOptions(el);
        } else {
          // 已成功设置，确保联动触发一次（有些情况下 before==after 不触发）
          el.dispatchEvent(new Event("change", { bubbles: true }));
        }
      } else {
        el.value = String(v);
      }
    } catch {}
  });

  root.querySelectorAll('select[data-os-persist]').forEach((sel) => {
    try { sel.dispatchEvent(new Event("change", { bubbles: true })); } catch {}
  });

  return cfg;
}

function __osBindPersistedFields(root = document) {
  let cfg = __osLoadConfig();

  const nodes = root.querySelectorAll("[data-os-persist]");
  nodes.forEach((el) => {
    const key = el.getAttribute("data-os-persist");
    if (!key) return;

    if (el.__osPersistBound) return;
    el.__osPersistBound = true;

    const handler = () => {
      const tag = (el.tagName || "").toLowerCase();
      const type = String(el.type || "").toLowerCase();

      let v;
      if (type === "checkbox") v = !!el.checked;
      else if (tag === "select") v = String(el.value ?? "");
      else v = String(el.value ?? "");

      __osSetByPath(cfg, key, v);
      __osSaveConfig(cfg);
    };

    el.addEventListener("input", handler);
    el.addEventListener("change", handler);
  });

  return {
    getConfig: () => (cfg = __osLoadConfig()),
    clear: () => {
      __OS_PERSIST_STORAGE.removeItem(__OS_PERSIST_KEY);
      cfg = {};
    },
    saveNow: () => __osSaveConfig(cfg),
  };
}

function __osInitPersistSidebarConfig() {
  __osHydratePersistedFields(document);
  window.OPENSTORYLINE_PERSIST = __osBindPersistedFields(document); // 可选：调试用
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", __osInitPersistSidebarConfig);
} else {
  __osInitPersistSidebarConfig();
}
