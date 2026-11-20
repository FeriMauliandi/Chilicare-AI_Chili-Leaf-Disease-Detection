// Regression test untuk sanitasi XSS di frontend/script.js
// Jalankan dengan: node tests/frontend/sanitizer.test.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

// Stub API browser seminimal mungkin agar script.js bisa dieksekusi di Node.
global.window = { location: { hostname: "localhost", hash: "" } };
global.localStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
};
global.document = {
  readyState: "complete",
  addEventListener: () => {},
  getElementById: () => null,
  querySelectorAll: () => [],
};

const scriptPath = path.join(__dirname, "..", "..", "frontend", "script.js");
vm.runInThisContext(fs.readFileSync(scriptPath, "utf8"), { filename: "script.js" });

assert.equal(typeof parseMarkdown, "function", "parseMarkdown harus tersedia");

// Satu-satunya tag yang boleh dihasilkan oleh parseMarkdown sendiri.
const ALLOWED_TAG = /<(\/?)(div|span|table|tr|th|td|h2|h3|hr|strong|em|br)\b[^>]*>/gi;

function assertNoInjectedMarkup(input) {
  const html = parseMarkdown(input);
  const leftover = html.replace(ALLOWED_TAG, "");
  assert.ok(
    !/[<>]/.test(leftover),
    `Markup berbahaya lolos dari escaping untuk input: ${JSON.stringify(input)}\nHasil: ${html}`,
  );
  return html;
}

// 1. Vektor XSS umum harus di-escape menjadi teks biasa.
const xssPayloads = [
  '<img src=x onerror="alert(1)">',
  "<script>alert('xss')</script>",
  '<a href="javascript:alert(1)">klik</a>',
  '" onmouseover="alert(1)',
  "<svg/onload=alert(1)>",
  "<iframe src=javascript:alert(1)></iframe>",
];
for (const payload of xssPayloads) {
  const html = assertNoInjectedMarkup(payload);
  assert.ok(!/<(script|img|a|iframe|svg)\b/i.test(html), "Tag mentah tidak boleh muncul");
}

// 2. Karakter khusus harus ter-escape, bukan hilang.
assert.equal(parseMarkdown("5 > 3 & 2 < 4"), "5 &gt; 3 &amp; 2 &lt; 4");
assert.ok(parseMarkdown('"quoted"').includes("&quot;quoted&quot;"));

// 3. Markdown yang sah tetap dirender.
assert.ok(parseMarkdown("**tebal**").includes("<strong"));
assert.ok(parseMarkdown("teks *miring*").includes("<em"));
assert.ok(parseMarkdown("## Judul").includes("<h2"));
assert.ok(parseMarkdown("| A | B |\n|---|---|\n| 1 | 2 |").includes("<table"));
assert.ok(parseMarkdown("1. satu\n2. dua").includes("flex gap-2"));

console.log("Frontend sanitizer tests: semua pengecekan lulus.");
