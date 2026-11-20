// const API_DETECT_URL = "http://localhost:8000/detect";
// const API_ASK_URL = "http://localhost:8000/ask";

// const API_DETECT_URL = "https://r7sc5m17-8000.asse.devtunnels.ms/detect";
// const API_ASK_URL = "https://r7sc5m17-8000.asse.devtunnels.ms/ask";

// Endpoint dinamis: gunakan relative path jika di-host di HF Space / localhost, atau URL absolut sebagai fallback
const IS_HF_OR_LOCAL = typeof window !== "undefined" && (
  window.location.hostname.includes("hf.space") ||
  window.location.hostname.includes("huggingface.co") ||
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
);
const API_BASE = IS_HF_OR_LOCAL ? "" : "https://feryms-chilicareai.hf.space";
const API_DETECT_URL = `${API_BASE}/detect`;
const API_ASK_URL = `${API_BASE}/ask`;

let selectedFile = null;
let chatHistory = [];
const CHAT_STORAGE_KEY = "chilicare_chat_history";

function parseMarkdown(text) {
  if (!text) return "";
  let html = text;
  html = html.replace(/(?:^\|.*\|(?:\n|\r|$))+/gm, function (match) {
    let rows = match.trim().split("\n");
    let tableHtml =
      '<div class="overflow-x-auto my-5 rounded-xl ring-1 ring-slate-200 shadow-sm"><table class="w-full text-sm text-left text-slate-600">';

    rows.forEach((row, index) => {
      if (row.match(/^\|[\s\-\:]+\|/)) return;

      let cleanRow = row.replace(/^\||\|$/g, "");
      let cols = cleanRow.split("|").map((c) => c.trim());

      tableHtml +=
        '<tr class="border-b border-slate-200 last:border-0 hover:bg-slate-50 transition-colors">';
      cols.forEach((col) => {
        if (index === 0) {
          tableHtml += `<th class="px-4 py-3 bg-slate-100 font-semibold text-slate-700 whitespace-nowrap">${col}</th>`;
        } else {
          tableHtml += `<td class="px-4 py-3 align-top">${col}</td>`;
        }
      });
      tableHtml += "</tr>";
    });
    tableHtml += "</table></div>";
    return tableHtml;
  });

  // 1. Headings (H2, H3, H4)
  html = html.replace(
    /^##\s+(.*$)/gim,
    '<h2 class="text-xl font-bold text-slate-800 mt-5 mb-2">$1</h2>',
  );
  html = html.replace(
    /^###\s+(.*$)/gim,
    '<h3 class="text-lg font-bold text-slate-800 mt-5 mb-2">$1</h3>',
  );
  html = html.replace(
    /^####\s+(.*$)/gim,
    '<h3 class="text-lg font-bold text-slate-800 mt-5 mb-2">$1</h3>',
  );

  // 2. Garis Pembatas (Horizontal Rule)
  html = html.replace(/^---$/gm, '<hr class="my-4 border-slate-200" />');

  // 3. Bold & Italic
  html = html.replace(
    /\*\*(.*?)\*\*/g,
    '<strong class="font-bold text-slate-800">$1</strong>',
  );
  html = html.replace(
    /(?<!^)\*(.*?)\*/g,
    '<em class="italic text-slate-700">$1</em>',
  );

  // 4. Bullet Points (Menangkap *, -, dan •)
  html = html.replace(
    /^[\*\-•]\s+(.*$)/gim,
    '<div class="flex gap-2 mt-1.5"><span class="text-rose-500 font-bold shrink-0">•</span><span>$1</span></div>',
  );

  // 5. Numbered List (Menangkap 1., 2., 3., dst) agar rapi sejajar
  html = html.replace(
    /^(\d+)\.\s+(.*$)/gim,
    '<div class="flex gap-2 mt-1.5"><span class="text-rose-500 font-bold shrink-0">$1.</span><span>$2</span></div>',
  );

  // 6. Ubah Enter menjadi <br/>
  html = html.replace(/\n/g, "<br/>");

  // 7. PEMBERSIHAN EKSTREM: Hapus <br/> yang menumpuk di sekitar elemen UI
  // Bersihkan sekitar garis pembatas
  html = html.replace(/(<br\/>)+<hr/g, "<hr");
  html = html.replace(/<hr(.*?)>(<br\/>)+/g, "<hr$1>");

  // Bersihkan sekitar Headings
  html = html.replace(/(<br\/>)+<h2/g, "<h2");
  html = html.replace(/<\/h2>(<br\/>)+/g, "</h2>");
  html = html.replace(/(<br\/>)+<h3/g, "<h3");
  html = html.replace(/<\/h3>(<br\/>)+/g, "</h3>");

  // Bersihkan sekitar kotak list (bullet & angka)
  html = html.replace(/(<br\/>)+<div/g, "<div");
  html = html.replace(/<\/div>(<br\/>)+/g, "</div>");

  // Maksimal 2 <br/> berturut-turut untuk paragraf biasa
  html = html.replace(/(<br\/>){3,}/g, "<br/><br/>");

  return html;
}

function switchMenu(menu) {
  const secDetect = document.getElementById("section-detect");
  const secChat = document.getElementById("section-chat");
  const btnDetect = document.getElementById("btn-menu-detect");
  const btnChat = document.getElementById("btn-menu-chat");

  const activeClass =
    "flex-1 md:w-full flex items-center justify-center md:justify-start gap-2 md:gap-3 px-4 py-2.5 md:py-3 rounded-xl font-medium transition-all bg-rose-50 text-rose-600 text-sm md:text-base whitespace-nowrap shadow-sm md:shadow-none";
  const inactiveClass =
    "flex-1 md:w-full flex items-center justify-center md:justify-start gap-2 md:gap-3 px-4 py-2.5 md:py-3 rounded-xl font-medium transition-all text-slate-500 hover:bg-slate-50 hover:text-slate-700 text-sm md:text-base whitespace-nowrap";

  if (menu === "detect") {
    secDetect.classList.remove("hidden");
    secChat.classList.add("hidden");
    secChat.classList.remove("flex");
    btnDetect.className = activeClass;
    btnChat.className = inactiveClass;
  } else {
    secDetect.classList.add("hidden");
    secChat.classList.remove("hidden");
    secChat.classList.add("flex");
    btnChat.className = activeClass;
    btnDetect.className = inactiveClass;
    const chatBox = document.getElementById("chat-messages");
    if (chatBox) scrollToBottom(chatBox, false);
  }
}

function handleFileChange(event) {
  const file = event.target.files[0];
  if (file) {
    selectedFile = file;

    // 1. Tampilkan nama file & tombol analisis
    const fileNameEl = document.getElementById("file-name");
    if (fileNameEl) {
      fileNameEl.textContent = `File: ${file.name}`;
      fileNameEl.classList.remove("hidden");
    }
    document.getElementById("btn-upload").classList.remove("hidden");

    // 2. KONTROL TAMPILAN KOTAK UPLOAD
    const imagePreview = document.getElementById("image-preview");
    const uploadPlaceholder = document.getElementById("upload-placeholder");

    // Masukkan sumber gambar
    imagePreview.src = URL.createObjectURL(file);

    // Sembunyikan ikon/teks, lalu tampilkan gambarnya
    if (uploadPlaceholder) uploadPlaceholder.classList.add("hidden");
    if (imagePreview) imagePreview.classList.remove("hidden");

    // 3. Reset hasil deteksi sebelumnya jika ada
    const detectResult = document.getElementById("detect-result");
    const detectError = document.getElementById("detect-error");
    if (detectResult) detectResult.classList.add("hidden");
    if (detectError) detectError.classList.add("hidden");
  }
}

async function handleUpload() {
  if (!selectedFile) return;

  const btnUpload = document.getElementById("btn-upload");
  const errorContainer = document.getElementById("detect-error");
  const resultContainer = document.getElementById("detect-result");

  btnUpload.disabled = true;
  btnUpload.textContent = "Memproses...";
  errorContainer.classList.add("hidden");
  resultContainer.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(API_DETECT_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error(`Error server: ${response.status}`);

    const data = await response.json();

    // Tampilkan hasil deteksi (gambar ber-bounding box dan teksnya)
    tampilkanHasilDeteksi(data);

    // --- KODE RESET KOTAK UPLOAD DITAMBAHKAN DI SINI ---
    // 1. Sembunyikan gambar preview daun dan tampilkan kembali ikon placeholder
    const imagePreview = document.getElementById("image-preview");
    const uploadPlaceholder = document.getElementById("upload-placeholder");

    if (imagePreview) imagePreview.classList.add("hidden");
    if (uploadPlaceholder) uploadPlaceholder.classList.remove("hidden");

    // 2. Sembunyikan teks nama file dan tombol analisis agar bersih
    document.getElementById("file-name").classList.add("hidden");
    btnUpload.classList.add("hidden");

    // 3. Bersihkan memori agar user bisa mengunggah file yang sama lagi jika perlu
    selectedFile = null;
    document.getElementById("file-upload").value = "";
    // --------------------------------------------------
  } catch (err) {
    errorContainer.textContent = "Gagal terhubung ke server.";
    errorContainer.classList.remove("hidden");
  } finally {
    btnUpload.disabled = false;
    btnUpload.textContent = "Mulai Analisis AI";
  }
}

function tampilkanHasilDeteksi(data) {
  const resultContainer = document.getElementById("detect-result");
  const resultImage = document.getElementById("result-image");
  const resultDetails = document.getElementById("result-details");

  if (data.image_base64) {
    resultImage.src = `data:image/jpeg;base64,${data.image_base64}`;
  }

  resultDetails.innerHTML = "";
  if (data.total_detections > 0) {
    data.results.forEach((item) => {
      const confidencePercent = (item.confidence * 100).toFixed(0);

      // MENGGUNAKAN FUNGSI PARSE MARKDOWN
      const formattedNarrative = parseMarkdown(item.narrative);

      const div = document.createElement("div");
      div.className =
        "bg-white ring-1 ring-slate-200 shadow-sm rounded-2xl p-4 md:p-6 relative overflow-hidden";

      div.innerHTML = `
                <div class="absolute top-0 left-0 w-1 md:w-1.5 h-full bg-rose-500"></div>
                <h4 class="text-base md:text-lg font-bold text-slate-800 mb-2 flex flex-col md:flex-row md:items-center justify-between gap-1">
                    <span>${item.class}</span>
                    <span class="text-xs font-semibold bg-slate-100 text-slate-500 px-2 py-1 rounded-md w-fit">Keyakinan: ${confidencePercent}%</span>
                </h4>
                <div class="text-slate-600 text-xs md:text-sm leading-relaxed mt-2 md:mt-3">
                    ${formattedNarrative}
                </div>
            `;
      resultDetails.appendChild(div);
    });
  } else {
    resultDetails.innerHTML = `
            <div class="bg-emerald-50 text-emerald-700 p-4 rounded-2xl ring-1 ring-emerald-200 font-medium flex items-center gap-3 text-sm md:text-base">
                <span class="text-xl">✅</span> Tidak ada penyakit terdeteksi. Daun tampak sehat!
            </div>
        `;
  }

  resultContainer.classList.remove("hidden");

  setTimeout(() => {
    resultContainer.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 500);
}

async function handleSendChat(event) {
  event.preventDefault();

  const inputEl = document.getElementById("chat-input");
  const btnSend = document.getElementById("btn-chat-send");
  const question = inputEl.value.trim();
  if (!question) return;

  const chatEmpty = document.getElementById("chat-empty");
  if (chatEmpty) chatEmpty.classList.add("hidden");

  // 1. Tampilkan pertanyaan User
  appendMessage("user", question);
  inputEl.value = "";
  inputEl.disabled = true;
  btnSend.disabled = true;

  // Salin riwayat percakapan sebelumnya (maksimal 10 putaran terakhir)
  const historyToSend = chatHistory.slice(-10);

  // 2. Tampilkan animasi loading (...) sebelum AI mulai mengetik
  const loadingId = showChatLoading();

  try {
    const response = await fetch(API_ASK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        history: historyToSend,
      }),
    });

    if (!response.ok) throw new Error("Gagal");

    // AI mulai membalas, hilangkan animasi loading
    removeChatLoading(loadingId);

    const aiBubble = appendMessage("assistant", "");
    const chatBox = document.getElementById("chat-messages");

    // 4. LOGIKA STREAMING DENGAN "REM MESIN TIK" (TYPEWRITER EFFECT)
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    
    let fullTextFromAPI = ""; // Tangki penampung teks super cepat dari server
    let charIndex = 0;

    // A. Fungsi pengetik independen (Jalan di latar belakang)
    // Angka 15 adalah kecepatan ketik (15 milidetik per batch karakter).
    const typingInterval = setInterval(() => {
      // Jika masih ada huruf di tangki yang belum ditampilkan
      if (charIndex < fullTextFromAPI.length) {
        // Ambil karakter yang tersedia secara presisi tanpa melompati buffer yang belum terbaca
        charIndex = Math.min(charIndex + 4, fullTextFromAPI.length);
        
        aiBubble.innerHTML = parseMarkdown(fullTextFromAPI.slice(0, charIndex));
        scrollToBottom(chatBox);
      }
    }, 15); 

    // B. Pipa penyedot dari API Server
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        // Matikan interval pengetik HANYA JIKA semua teks sudah berhasil diketik ke layar
        const waitComplete = setInterval(() => {
          if (charIndex >= fullTextFromAPI.length) {
            clearInterval(typingInterval);
            clearInterval(waitComplete);

            // Pastikan teks akhir ter-render utuh sesuai respons server
            aiBubble.innerHTML = parseMarkdown(fullTextFromAPI);
            scrollToBottom(chatBox);

            // Simpan percakapan ke chatHistory lokal setelah streaming sukses
            chatHistory.push({ role: "user", content: question });
            chatHistory.push({ role: "assistant", content: fullTextFromAPI });
            saveChatHistory();
          }
        }, 30);
        break;
      }

      // Terjemahkan byte dan masukkan langsung ke tangki penampung
      const chunkText = decoder.decode(value, { stream: true });
      fullTextFromAPI += chunkText; 
    }

  } catch (err) {
    removeChatLoading(loadingId);
    appendMessage("assistant", "❌ Gagal terhubung ke server. Silakan coba lagi.");
  } finally {
    inputEl.disabled = false;
    btnSend.disabled = false;
    inputEl.focus();
  }
}

function saveChatHistory() {
  try {
    // Simpan maksimal 50 pesan terakhir agar efisien dan hemat memori
    if (chatHistory.length > 50) {
      chatHistory = chatHistory.slice(-50);
    }
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chatHistory));
  } catch (e) {
    console.warn("Gagal menyimpan riwayat chat ke localStorage:", e);
  }
}

function loadChatHistory() {
  try {
    const saved = localStorage.getItem(CHAT_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length > 0) {
        chatHistory = parsed;
        renderSavedChatHistory();
      }
    }
  } catch (e) {
    console.warn("Gagal memuat riwayat chat dari localStorage:", e);
    chatHistory = [];
  }
}

function renderSavedChatHistory() {
  const chatBox = document.getElementById("chat-messages");
  if (!chatBox) return;

  if (chatHistory && chatHistory.length > 0) {
    const chatEmpty = document.getElementById("chat-empty");
    if (chatEmpty) chatEmpty.classList.add("hidden");

    chatHistory.forEach((msg) => {
      appendMessage(msg.role, msg.content);
    });

    scrollToBottom(chatBox, false);
  }
}

function clearChatHistory() {
  chatHistory = [];
  try {
    localStorage.removeItem(CHAT_STORAGE_KEY);
  } catch (e) {
    console.warn("Gagal menghapus riwayat chat dari localStorage:", e);
  }
  const chatBox = document.getElementById("chat-messages");
  if (!chatBox) return;
  chatBox.innerHTML = `
    <div
      id="chat-empty"
      class="h-full flex flex-col items-center justify-center text-[#6e8074] gap-4 py-8"
    >
      <div class="w-14 h-14 rounded-2xl bg-[#eaf4ed] border border-[#d9eee1] flex items-center justify-center">
        <svg class="w-7 h-7 text-[#255d3e]" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>
        </svg>
      </div>
      <div class="text-center max-w-sm">
        <p class="text-sm font-bold text-[#101c14] mb-1">Mulai Konsultasi Tanaman</p>
        <p class="text-xs text-[#6e8074] leading-relaxed">
          Tanyakan seputar perawatan cabai rawit, cara meracik fungisida nabati, atau gejala penyakit daun.
        </p>
      </div>
    </div>
  `;
}

function appendMessage(role, content) {
  const chatBox = document.getElementById("chat-messages");
  const wrapper = document.createElement("div");
  wrapper.className = `flex ${role === "user" ? "justify-end" : "justify-start"}`;

  const bubble = document.createElement("div");
  if (role === "user") {
    bubble.className =
      "max-w-[90%] md:max-w-[80%] p-3 md:p-4 rounded-2xl rounded-tr-sm bg-rose-500 text-white shadow-md shadow-rose-200 leading-relaxed text-sm md:text-[15px]";
    bubble.textContent = content;
  } else {
    bubble.className =
      "max-w-[90%] md:max-w-[80%] p-3 md:p-4 rounded-2xl rounded-tl-sm bg-white ring-1 ring-slate-200 text-slate-700 shadow-sm leading-relaxed text-sm md:text-[15px]";

    // Jika ada konten (user history), parse markdown. Jika kosong (awal streaming), biarkan kosong.
    bubble.innerHTML = content ? parseMarkdown(content) : "";
  }

  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  scrollToBottom(chatBox);

  // KUNCI PERUBAHAN: Kembalikan elemen bubble agar bisa di-update oleh fungsi streaming
  return bubble;
}

function showChatLoading() {
  const chatBox = document.getElementById("chat-messages");
  const wrapper = document.createElement("div");
  const id = "loading-" + Date.now();
  wrapper.id = id;
  wrapper.className = `flex justify-start`;
  wrapper.innerHTML = `
        <div class="bg-white ring-1 ring-slate-200 text-slate-500 p-4 md:p-5 rounded-2xl rounded-tl-sm shadow-sm flex items-center">
            <div class="dot-flashing ml-2"></div>
        </div>
    `;
  chatBox.appendChild(wrapper);
  scrollToBottom(chatBox);
  return id;
}

function removeChatLoading(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom(element, smooth = true) {
  if (!element) return;
  element.scrollTo({
    top: element.scrollHeight,
    behavior: smooth ? "smooth" : "auto",
  });
}

// Inisialisasi pemuatan riwayat chat yang tersimpan di localStorage
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", loadChatHistory);
} else {
  loadChatHistory();
}

