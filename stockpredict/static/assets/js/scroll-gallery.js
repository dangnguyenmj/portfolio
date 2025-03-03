document.querySelector("html").classList.contains("is-builder") || function() {
    const n = document.querySelectorAll(".gallery-wrapper");
    n.length && n.forEach((n => {
        const t = n.querySelector(".grid-container-1")
          , e = n.querySelector(".grid-container-2")
          , i = n.querySelector(".grid-container-3")
          , o = t ? getComputedStyle(t).transform : null
          , r = e ? getComputedStyle(e).transform : null
          , s = i ? getComputedStyle(i).transform : null;
        function l() {
            const l = window.scrollY
              , a = window.innerHeight
              , d = n.getBoundingClientRect().top + window.scrollY - a;
            if (l >= d) {
                const m = o ? new DOMMatrix(o) : null
                  , w = r ? new DOMMatrix(r) : null
                  , h = s ? new DOMMatrix(s) : null;
                if (m) {
                    const f = t.querySelectorAll(".grid-item")
                      , g = f.length;
                    function c() {
                        for (let n = 0; n < g; n++) {
                            const e = f[n].cloneNode(!0);
                            t.appendChild(e)
                        }
                    }
                    switch (f.forEach((n => {
                        n.querySelector("img").classList.remove("hidden")
                    }
                    )),
                    g >= 8 && !t.classList.contains("moving-left") && (m.m41 = -2e3),
                    g >= 8 && !t.classList.contains("moving-left") && window.innerWidth < 1200 && (m.m41 = -900),
                    !0) {
                    case g < 8:
                        c();
                    case t.classList.contains("moving-right"):
                        translateX1 = m.m41 + .9 * (l - d);
                        break;
                    case t.classList.contains("moving-left"):
                        translateX1 = m.m41 - .9 * (l - d);
                        break;
                    default:
                        window.innerWidth >= 2560 ? translateX1 = m.m41 + .9 * (l - d) : window.innerWidth >= 1440 ? translateX1 = m.m41 + .55 * (l - d) : window.innerWidth >= 1024 ? translateX1 = m.m41 + .45 * (l - d) : window.innerWidth >= 768 ? translateX1 = m.m41 + .35 * (l - d) : window.innerWidth >= 320 && (translateX1 = m.m41 + .25 * (l - d))
                    }
                    t.style.transform = `translate3d(${translateX1}px, 0, 0)`
                }
                if (w) {
                    const u = e.querySelectorAll(".grid-item")
                      , W = u.length;
                    function c() {
                        for (let n = 0; n < W; n++) {
                            const t = u[n].cloneNode(!0);
                            e.appendChild(t)
                        }
                    }
                    switch (u.forEach((n => {
                        n.querySelector("img").classList.remove("hidden")
                    }
                    )),
                    W >= 8 && e.classList.contains("moving-left") && (w.m41 = -2e3),
                    W >= 8 && !e.classList.contains("moving-left") && window.innerWidth < 1200 && (w.m41 = 0),
                    !0) {
                    case W < 8:
                        c();
                    case e.classList.contains("moving-right"):
                        translateX2 = w.m41 + .9 * (l - d);
                        break;
                    case e.classList.contains("moving-left"):
                        translateX2 = w.m41 - .9 * (l - d);
                        break;
                    default:
                        window.innerWidth >= 2560 ? translateX2 = w.m41 - .9 * (l - d) : window.innerWidth >= 1440 ? translateX2 = w.m41 - .55 * (l - d) : window.innerWidth >= 1024 ? translateX2 = w.m41 - .45 * (l - d) : window.innerWidth >= 768 ? translateX2 = w.m41 - .35 * (l - d) : window.innerWidth >= 320 && (translateX2 = w.m41 - .25 * (l - d))
                    }
                    e.style.transform = `translate3d(${translateX2}px, 0, 0)`
                }
                if (h) {
                    const L = i.querySelectorAll(".grid-item")
                      , y = L.length;
                    L.forEach((n => {
                        n.querySelector("img").classList.remove("hidden")
                    }
                    )),
                    y >= 8 && !i.classList.contains("moving-left") && (h.m41 = -2e3),
                    y >= 8 && !i.classList.contains("moving-left") && window.innerWidth < 1200 && (h.m41 = -900);
                    let v = h.m41 + .9 * (l - d);
                    function c() {
                        for (let n = 0; n < y; n++) {
                            const t = L[n].cloneNode(!0);
                            i.appendChild(t)
                        }
                    }
                    y < 8 && c(),
                    i.classList.contains("moving-left") ? i.classList.contains("moving-left") && (y < 8 && c(),
                    window.innerWidth >= 2560 ? v = h.m41 - .9 * (l - d) : window.innerWidth >= 1440 ? v = h.m41 - .55 * (l - d) : window.innerWidth >= 1024 ? v = h.m41 - .45 * (l - d) : window.innerWidth >= 768 ? v = h.m41 - .35 * (l - d) : window.innerWidth >= 320 && (v = h.m41 - .25 * (l - d))) : window.innerWidth >= 2560 ? v = h.m41 + .9 * (l - d) : window.innerWidth >= 1440 ? v = h.m41 + .55 * (l - d) : window.innerWidth >= 1024 ? v = h.m41 + .45 * (l - d) : window.innerWidth >= 768 ? v = h.m41 + .35 * (l - d) : window.innerWidth >= 320 && (v = h.m41 + .25 * (l - d)),
                    i.style.transform = `translate3d(${v}px, 0, 0)`
                }
            }
        }
        window.addEventListener("DOMContentLoaded", ( () => {
            requestAnimationFrame(l)
        }
        )),
        window.addEventListener("scroll", ( () => {
            requestAnimationFrame(l)
        }
        ))
    }
    ))
}();
