/* =====================================================================
   AI-Text Forensics — site.js
   One vanilla-JS file powering: canvas constellation, scroll reveals,
   animated counters, tabbed results, figure lightbox, nav progress +
   active section dots, copy-BibTeX, card tilt.
   Defensive: every feature no-ops when its elements are absent.
   All motion respects prefers-reduced-motion. No dependencies.
   ===================================================================== */
(function () {
  "use strict";

  document.documentElement.classList.add("js");

  function $(sel, ctx) { return (ctx || document).querySelector(sel); }
  function $$(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }

  var motionQuery = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  function reduced() { return !!(motionQuery && motionQuery.matches); }
  function onMotionChange(fn) {
    if (motionQuery && motionQuery.addEventListener) motionQuery.addEventListener("change", fn);
  }

  /* ---------------------------------------------------------------
     1. Particle constellation (index hero canvas).
        Drifting points joined by proximity lines; paused under
        prefers-reduced-motion and while the tab is hidden.
     --------------------------------------------------------------- */
  function initConstellation() {
    var canvas = document.getElementById("constellation");
    if (!canvas || !canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    if (!ctx) return;

    var pts = [], raf = 0, running = false, w = 0, h = 0;
    var rgb = "124,116,255";

    function readColor() {
      var v = getComputedStyle(document.body).getPropertyValue("--particle").trim();
      if (v) rgb = v;
    }
    function resize() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth; h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      readColor();
      seed();
    }
    function seed() {
      var n = Math.max(28, Math.min(110, Math.round(w * h / 16000)));
      pts = [];
      for (var i = 0; i < n; i++) {
        pts.push({
          x: Math.random() * w, y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.24, vy: (Math.random() - 0.5) * 0.24,
          r: Math.random() * 1.5 + 0.6
        });
      }
    }
    function frame() {
      ctx.clearRect(0, 0, w, h);
      var link = 120, i, j, p, q, dx, dy, d2;
      for (i = 0; i < pts.length; i++) {
        p = pts[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < -20) p.x = w + 20; else if (p.x > w + 20) p.x = -20;
        if (p.y < -20) p.y = h + 20; else if (p.y > h + 20) p.y = -20;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, 6.2832);
        ctx.fillStyle = "rgba(" + rgb + ",.55)";
        ctx.fill();
      }
      for (i = 0; i < pts.length; i++) {
        for (j = i + 1; j < pts.length; j++) {
          p = pts[i]; q = pts[j];
          dx = p.x - q.x; dy = p.y - q.y; d2 = dx * dx + dy * dy;
          if (d2 < link * link) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
            ctx.strokeStyle = "rgba(" + rgb + "," +
              ((1 - Math.sqrt(d2) / link) * 0.22).toFixed(3) + ")";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }
    }
    function loop() { frame(); raf = window.requestAnimationFrame(loop); }
    function start() {
      if (running || reduced() || document.hidden) return;
      running = true; raf = window.requestAnimationFrame(loop);
    }
    function stop() {
      running = false;
      if (raf) { window.cancelAnimationFrame(raf); raf = 0; }
    }

    document.addEventListener("visibilitychange", function () {
      if (document.hidden) stop(); else start();
    });
    onMotionChange(function () {
      if (reduced()) { stop(); frame(); } else { start(); }
    });
    var rT;
    window.addEventListener("resize", function () {
      clearTimeout(rT);
      rT = setTimeout(function () { resize(); if (!running) frame(); }, 150);
    });

    resize();
    if (reduced()) frame(); else start(); /* static frame when motion is off */
  }

  /* ---------------------------------------------------------------
     2. Scroll reveal with stagger ([data-reveal], [data-stagger]).
     --------------------------------------------------------------- */
  function initReveal() {
    $$("[data-stagger]").forEach(function (group) {
      Array.prototype.forEach.call(group.children, function (child, i) {
        if (!child.hasAttribute("data-reveal")) child.setAttribute("data-reveal", "");
        child.style.setProperty("--d", (Math.min(i, 7) * 0.07).toFixed(2) + "s");
      });
    });
    var els = $$("[data-reveal]");
    if (!els.length) return;
    if (!("IntersectionObserver" in window) || reduced()) {
      els.forEach(function (el) { el.classList.add("in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -40px 0px", threshold: 0.1 });
    els.forEach(function (el) { io.observe(el); });
  }

  /* ---------------------------------------------------------------
     3. Animated counters ([data-count], optional [data-suffix]).
        Markup already contains the final value, so no JS = no loss.
     --------------------------------------------------------------- */
  function initCounters() {
    var els = $$("[data-count]");
    if (!els.length) return;

    function paintFinal(el, raw, dec, suffix) {
      el.textContent = parseFloat(raw).toFixed(dec) + suffix;
    }
    function animate(el) {
      var raw = el.getAttribute("data-count");
      var target = parseFloat(raw);
      if (isNaN(target)) return;
      var dec = (raw.split(".")[1] || "").length;
      var suffix = el.getAttribute("data-suffix") || "";
      if (reduced()) { paintFinal(el, raw, dec, suffix); return; }
      var t0 = null, dur = 1300;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var t = Math.min(1, (ts - t0) / dur);
        var eased = 1 - Math.pow(1 - t, 3);
        el.textContent = (target * eased).toFixed(dec) + suffix;
        if (t < 1) window.requestAnimationFrame(step);
        else paintFinal(el, raw, dec, suffix);
      }
      window.requestAnimationFrame(step);
    }
    if (!("IntersectionObserver" in window)) return; /* markup already final */
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          io.unobserve(entry.target);
          animate(entry.target);
        }
      });
    }, { threshold: 0.4 });
    els.forEach(function (el) { io.observe(el); });
  }

  /* ---------------------------------------------------------------
     4. Tabs ([data-tabs] > [role=tab] / [role=tabpanel]).
     --------------------------------------------------------------- */
  function initTabs() {
    $$("[data-tabs]").forEach(function (root) {
      var tabs = $$('[role="tab"]', root);
      var panels = $$('[role="tabpanel"]', root);
      if (!tabs.length || !panels.length) return;

      function select(tab, focus) {
        tabs.forEach(function (t) {
          var on = t === tab;
          t.setAttribute("aria-selected", on ? "true" : "false");
          t.tabIndex = on ? 0 : -1;
          var p = document.getElementById(t.getAttribute("aria-controls") || "");
          if (p) p.hidden = !on;
        });
        if (focus) tab.focus();
      }
      tabs.forEach(function (t, i) {
        t.addEventListener("click", function () { select(t); });
        t.addEventListener("keydown", function (ev) {
          var d = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
          if (!d) return;
          ev.preventDefault();
          select(tabs[(i + d + tabs.length) % tabs.length], true);
        });
      });
      var initial = tabs.filter(function (t) {
        return t.getAttribute("aria-selected") === "true";
      })[0] || tabs[0];
      select(initial);
    });
  }

  /* ---------------------------------------------------------------
     5. Lightbox for figure images (.fig-zoom img / .figure-card img).
     --------------------------------------------------------------- */
  function initLightbox() {
    var imgs = $$(".figure-card img, .fig-zoom img, .project-card > img");
    if (!imgs.length) return;

    var lb = document.createElement("div");
    lb.className = "lightbox";
    lb.hidden = true;
    lb.setAttribute("role", "dialog");
    lb.setAttribute("aria-modal", "true");
    lb.setAttribute("aria-label", "Figure viewer");
    lb.innerHTML =
      '<div class="lb-backdrop"></div>' +
      '<figure class="lb-body glass">' +
      '  <img alt="">' +
      '  <figcaption class="lb-caption"></figcaption>' +
      '  <button type="button" class="lb-close" aria-label="Close figure viewer">&times;</button>' +
      "</figure>";
    document.body.appendChild(lb);

    var lbImg = $("img", lb), lbCap = $(".lb-caption", lb);
    var lastFocus = null, closeTimer = 0;

    function open(img) {
      clearTimeout(closeTimer);
      lb.classList.remove("closing");
      lbImg.src = img.currentSrc || img.src;
      lbImg.alt = img.alt || "";
      var fig = img.closest("figure");
      var cap = fig ? $("figcaption", fig) : null;
      lbCap.innerHTML = cap ? cap.innerHTML : (img.alt || "");
      lastFocus = document.activeElement;
      lb.hidden = false;
      document.body.classList.add("no-scroll");
      $(".lb-close", lb).focus();
    }
    function teardown() {
      lb.hidden = true;
      document.body.classList.remove("no-scroll");
      lbImg.src = "";
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }
    function close() {
      if (lb.hidden || lb.classList.contains("closing")) return;
      if (reduced()) { teardown(); return; }
      /* play the close animation, then hide (animationend + timer fallback) */
      function finish() {
        clearTimeout(closeTimer);
        lb.removeEventListener("animationend", finish);
        if (!lb.classList.contains("closing")) return; /* reopened meanwhile */
        lb.classList.remove("closing");
        teardown();
      }
      lb.classList.add("closing");
      lb.addEventListener("animationend", finish);
      closeTimer = setTimeout(finish, 320);
    }

    imgs.forEach(function (img) {
      img.addEventListener("click", function () { open(img); });
      img.setAttribute("tabindex", "0");
      img.setAttribute("role", "button");
      img.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); open(img); }
      });
    });
    $(".lb-backdrop", lb).addEventListener("click", close);
    $(".lb-close", lb).addEventListener("click", close);
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && !lb.hidden) close();
    });
  }

  /* ---------------------------------------------------------------
     6. Sticky nav: scroll progress bar + active section dots.
     --------------------------------------------------------------- */
  function initNav() {
    var nav = $(".topnav");
    if (!nav) return;
    var bar = $(".nav-progress", nav);
    var dots = $$(".nav-dots a", nav);
    var secs = dots.map(function (d) {
      var hash = d.getAttribute("href") || "";
      return hash.charAt(0) === "#" ? document.getElementById(hash.slice(1)) : null;
    }).filter(Boolean);

    var ticking = false;
    function update() {
      ticking = false;
      var docEl = document.documentElement;
      var max = docEl.scrollHeight - window.innerHeight;
      if (bar) {
        var p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
        bar.style.transform = "scaleX(" + p.toFixed(4) + ")";
      }
      if (secs.length) {
        var mark = window.scrollY + window.innerHeight * 0.33;
        var act = secs[0];
        secs.forEach(function (s) { if (s.offsetTop <= mark) act = s; });
        dots.forEach(function (d) {
          d.classList.toggle("active", d.getAttribute("href") === "#" + act.id);
        });
      }
    }
    function request() {
      if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
    }
    window.addEventListener("scroll", request, { passive: true });
    window.addEventListener("resize", request);
    update();
  }

  /* ---------------------------------------------------------------
     7. Copy buttons ([data-copy="#selector"]) + toast.
     --------------------------------------------------------------- */
  var toastEl = null, toastTimer = 0;
  function toast(msg) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.className = "toast glass";
      toastEl.setAttribute("role", "status");
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove("show"); }, 2200);
  }
  function copyFallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* noop */ }
    document.body.removeChild(ta);
  }
  function initCopy() {
    $$("[data-copy]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var src = $(btn.getAttribute("data-copy"));
        if (!src) return;
        var text = src.textContent;
        var done = function () { toast(btn.getAttribute("data-copied-msg") || "Copied to clipboard"); };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {
            copyFallback(text); done();
          });
        } else {
          copyFallback(text); done();
        }
      });
    });
  }

  /* ---------------------------------------------------------------
     8. Card tilt on hover (.tilt) — pointer-fine, motion-safe only.
     --------------------------------------------------------------- */
  function initTilt() {
    if (reduced()) return;
    if (!window.matchMedia || !matchMedia("(hover: hover) and (pointer: fine)").matches) return;
    $$(".tilt").forEach(function (card) {
      var raf = 0;
      card.addEventListener("pointermove", function (e) {
        if (raf || reduced()) return;
        raf = window.requestAnimationFrame(function () {
          raf = 0;
          var r = card.getBoundingClientRect();
          var rx = ((e.clientY - r.top) / r.height - 0.5) * -3.5;
          var ry = ((e.clientX - r.left) / r.width - 0.5) * 4.5;
          card.style.transform = "perspective(900px) rotateX(" + rx.toFixed(2) +
            "deg) rotateY(" + ry.toFixed(2) + "deg) translateY(-3px)";
        });
      });
      card.addEventListener("pointerleave", function () {
        if (raf) { window.cancelAnimationFrame(raf); raf = 0; }
        card.style.transform = "";
      });
    });
  }

  /* ---------------------------------------------------------------
     9. Pointer glow trail on cards (.project-card / .figure-card).
        Sets --mx/--my used by a CSS radial-gradient background layer.
        Pointer-fine + motion-safe only; CSS falls back to no trail.
     --------------------------------------------------------------- */
  function initGlowTrail() {
    if (reduced()) return;
    if (!window.matchMedia || !matchMedia("(hover: hover) and (pointer: fine)").matches) return;
    $$(".project-card, .figure-card").forEach(function (card) {
      var raf = 0, px = 0, py = 0;
      card.addEventListener("pointermove", function (e) {
        px = e.clientX; py = e.clientY;
        if (raf || reduced()) return;
        raf = window.requestAnimationFrame(function () {
          raf = 0;
          var r = card.getBoundingClientRect();
          if (!r.width || !r.height) return;
          card.style.setProperty("--mx", ((px - r.left) / r.width * 100).toFixed(2) + "%");
          card.style.setProperty("--my", ((py - r.top) / r.height * 100).toFixed(2) + "%");
        });
      });
      card.addEventListener("pointerleave", function () {
        if (raf) { window.cancelAnimationFrame(raf); raf = 0; }
        card.style.removeProperty("--mx");
        card.style.removeProperty("--my");
      });
    });
  }

  /* ---------------------------------------------------------------
     boot
     --------------------------------------------------------------- */
  function init() {
    initConstellation();
    initReveal();
    initCounters();
    initTabs();
    initLightbox();
    initNav();
    initCopy();
    initTilt();
    initGlowTrail();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
