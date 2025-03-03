/*! @vimeo/player v2.14.1 | (c) 2020 Vimeo | MIT License | https://github.com/vimeo/player.js */
!function(e, t) {
    "object" == typeof exports && "undefined" != typeof module ? module.exports = t() : "function" == typeof define && define.amd ? define(t) : ((e = "undefined" != typeof globalThis ? globalThis : e || self).Vimeo = e.Vimeo || {},
    e.Vimeo.Player = t())
}(this, (function() {
    "use strict";
    function e(e, t) {
        for (var n = 0; n < t.length; n++) {
            var r = t[n];
            r.enumerable = r.enumerable || !1,
            r.configurable = !0,
            "value"in r && (r.writable = !0),
            Object.defineProperty(e, r.key, r)
        }
    }
    var t = "undefined" != typeof global && "[object global]" === {}.toString.call(global);
    function n(e, t) {
        return 0 === e.indexOf(t.toLowerCase()) ? e : "".concat(t.toLowerCase()).concat(e.substr(0, 1).toUpperCase()).concat(e.substr(1))
    }
    function r(e) {
        return /^(https?:)?\/\/((player|www)\.)?vimeo\.com(?=$|\/)/.test(e)
    }
    function o() {
        var e, t = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {}, n = t.id, o = t.url, i = n || o;
        if (!i)
            throw new Error("An id or url must be passed, either in an options object or as a data-vimeo-id or data-vimeo-url attribute.");
        if (e = i,
        !isNaN(parseFloat(e)) && isFinite(e) && Math.floor(e) == e)
            return "https://vimeo.com/".concat(i);
        if (r(i))
            return i.replace("http:", "https:");
        if (n)
            throw new TypeError("“".concat(n, "” is not a valid video id."));
        throw new TypeError("“".concat(i, "” is not a vimeo.com url."))
    }
    var i = void 0 !== Array.prototype.indexOf
      , a = "undefined" != typeof window && void 0 !== window.postMessage;
    if (!(t || i && a))
        throw new Error("Sorry, the Vimeo Player API is not available in this browser.");
    var u = "undefined" != typeof globalThis ? globalThis : "undefined" != typeof window ? window : "undefined" != typeof global ? global : "undefined" != typeof self ? self : {};
    /*!
   * weakmap-polyfill v2.0.1 - ECMAScript6 WeakMap polyfill
   * https://github.com/polygonplanet/weakmap-polyfill
   * Copyright (c) 2015-2020 Polygon Planet <polygon.planet.aqua@gmail.com>
   * @license MIT
   */
    !function(e) {
        if (!e.WeakMap) {
            var t = Object.prototype.hasOwnProperty
              , n = function(e, t, n) {
                Object.defineProperty ? Object.defineProperty(e, t, {
                    configurable: !0,
                    writable: !0,
                    value: n
                }) : e[t] = n
            };
            e.WeakMap = function() {
                function e() {
                    if (void 0 === this)
                        throw new TypeError("Constructor WeakMap requires 'new'");
                    if (n(this, "_id", "_WeakMap" + "_" + i() + "." + i()),
                    arguments.length > 0)
                        throw new TypeError("WeakMap iterable is not supported")
                }
                function o(e, n) {
                    if (!r(e) || !t.call(e, "_id"))
                        throw new TypeError(n + " method called on incompatible receiver " + typeof e)
                }
                function i() {
                    return Math.random().toString().substring(2)
                }
                return n(e.prototype, "delete", (function(e) {
                    if (o(this, "delete"),
                    !r(e))
                        return !1;
                    var t = e[this._id];
                    return !(!t || t[0] !== e) && (delete e[this._id],
                    !0)
                }
                )),
                n(e.prototype, "get", (function(e) {
                    if (o(this, "get"),
                    r(e)) {
                        var t = e[this._id];
                        return t && t[0] === e ? t[1] : void 0
                    }
                }
                )),
                n(e.prototype, "has", (function(e) {
                    if (o(this, "has"),
                    !r(e))
                        return !1;
                    var t = e[this._id];
                    return !(!t || t[0] !== e)
                }
                )),
                n(e.prototype, "set", (function(e, t) {
                    if (o(this, "set"),
                    !r(e))
                        throw new TypeError("Invalid value used as weak map key");
                    var i = e[this._id];
                    return i && i[0] === e ? (i[1] = t,
                    this) : (n(e, this._id, [e, t]),
                    this)
                }
                )),
                n(e, "_polyfill", !0),
                e
            }()
        }
        function r(e) {
            return Object(e) === e
        }
    }("undefined" != typeof self ? self : "undefined" != typeof window ? window : u);
    var l = function(e, t) {
        return e(t = {
            exports: {}
        }, t.exports),
        t.exports
    }((function(e) {
        /*! Native Promise Only
      v0.8.1 (c) Kyle Simpson
      MIT License: http://getify.mit-license.org
  */
        var t, n, r;
        r = function() {
            var e, t, n, r = Object.prototype.toString, o = "undefined" != typeof setImmediate ? function(e) {
                return setImmediate(e)
            }
            : setTimeout;
            try {
                Object.defineProperty({}, "x", {}),
                e = function(e, t, n, r) {
                    return Object.defineProperty(e, t, {
                        value: n,
                        writable: !0,
                        configurable: !1 !== r
                    })
                }
            } catch (t) {
                e = function(e, t, n) {
                    return e[t] = n,
                    e
                }
            }
            function i(e, r) {
                n.add(e, r),
                t || (t = o(n.drain))
            }
            function a(e) {
                var t, n = typeof e;
                return null == e || "object" != n && "function" != n || (t = e.then),
                "function" == typeof t && t
            }
            function u() {
                for (var e = 0; e < this.chain.length; e++)
                    l(this, 1 === this.state ? this.chain[e].success : this.chain[e].failure, this.chain[e]);
                this.chain.length = 0
            }
            function l(e, t, n) {
                var r, o;
                try {
                    !1 === t ? n.reject(e.msg) : (r = !0 === t ? e.msg : t.call(void 0, e.msg)) === n.promise ? n.reject(TypeError("Promise-chain cycle")) : (o = a(r)) ? o.call(r, n.resolve, n.reject) : n.resolve(r)
                } catch (e) {
                    n.reject(e)
                }
            }
            function c(e) {
                var t, n = this;
                if (!n.triggered) {
                    n.triggered = !0,
                    n.def && (n = n.def);
                    try {
                        (t = a(e)) ? i((function() {
                            var r = new d(n);
                            try {
                                t.call(e, (function() {
                                    c.apply(r, arguments)
                                }
                                ), (function() {
                                    s.apply(r, arguments)
                                }
                                ))
                            } catch (e) {
                                s.call(r, e)
                            }
                        }
                        )) : (n.msg = e,
                        n.state = 1,
                        n.chain.length > 0 && i(u, n))
                    } catch (e) {
                        s.call(new d(n), e)
                    }
                }
            }
            function s(e) {
                var t = this;
                t.triggered || (t.triggered = !0,
                t.def && (t = t.def),
                t.msg = e,
                t.state = 2,
                t.chain.length > 0 && i(u, t))
            }
            function f(e, t, n, r) {
                for (var o = 0; o < t.length; o++)
                    !function(o) {
                        e.resolve(t[o]).then((function(e) {
                            n(o, e)
                        }
                        ), r)
                    }(o)
            }
            function d(e) {
                this.def = e,
                this.triggered = !1
            }
            function h(e) {
                this.promise = e,
                this.state = 0,
                this.triggered = !1,
                this.chain = [],
                this.msg = void 0
            }
            function v(e) {
                if ("function" != typeof e)
                    throw TypeError("Not a function");
                if (0 !== this.__NPO__)
                    throw TypeError("Not a promise");
                this.__NPO__ = 1;
                var t = new h(this);
                this.then = function(e, n) {
                    var r = {
                        success: "function" != typeof e || e,
                        failure: "function" == typeof n && n
                    };
                    return r.promise = new this.constructor((function(e, t) {
                        if ("function" != typeof e || "function" != typeof t)
                            throw TypeError("Not a function");
                        r.resolve = e,
                        r.reject = t
                    }
                    )),
                    t.chain.push(r),
                    0 !== t.state && i(u, t),
                    r.promise
                }
                ,
                this.catch = function(e) {
                    return this.then(void 0, e)
                }
                ;
                try {
                    e.call(void 0, (function(e) {
                        c.call(t, e)
                    }
                    ), (function(e) {
                        s.call(t, e)
                    }
                    ))
                } catch (e) {
                    s.call(t, e)
                }
            }
            n = function() {
                var e, n, r;
                function o(e, t) {
                    this.fn = e,
                    this.self = t,
                    this.next = void 0
                }
                return {
                    add: function(t, i) {
                        r = new o(t,i),
                        n ? n.next = r : e = r,
                        n = r,
                        r = void 0
                    },
                    drain: function() {
                        var r = e;
                        for (e = n = t = void 0; r; )
                            r.fn.call(r.self),
                            r = r.next
                    }
                }
            }();
            var m = e({}, "constructor", v, !1);
            return v.prototype = m,
            e(m, "__NPO__", 0, !1),
            e(v, "resolve", (function(e) {
                return e && "object" == typeof e && 1 === e.__NPO__ ? e : new this((function(t, n) {
                    if ("function" != typeof t || "function" != typeof n)
                        throw TypeError("Not a function");
                    t(e)
                }
                ))
            }
            )),
            e(v, "reject", (function(e) {
                return new this((function(t, n) {
                    if ("function" != typeof t || "function" != typeof n)
                        throw TypeError("Not a function");
                    n(e)
                }
                ))
            }
            )),
            e(v, "all", (function(e) {
                var t = this;
                return "[object Array]" != r.call(e) ? t.reject(TypeError("Not an array")) : 0 === e.length ? t.resolve([]) : new t((function(n, r) {
                    if ("function" != typeof n || "function" != typeof r)
                        throw TypeError("Not a function");
                    var o = e.length
                      , i = Array(o)
                      , a = 0;
                    f(t, e, (function(e, t) {
                        i[e] = t,
                        ++a === o && n(i)
                    }
                    ), r)
                }
                ))
            }
            )),
            e(v, "race", (function(e) {
                var t = this;
                return "[object Array]" != r.call(e) ? t.reject(TypeError("Not an array")) : new t((function(n, r) {
                    if ("function" != typeof n || "function" != typeof r)
                        throw TypeError("Not a function");
                    f(t, e, (function(e, t) {
                        n(t)
                    }
                    ), r)
                }
                ))
            }
            )),
            v
        }
        ,
        (n = u)[t = "Promise"] = n[t] || r(),
        e.exports && (e.exports = n[t])
    }
    ))
      , c = new WeakMap;
    function s(e, t, n) {
        var r = c.get(e.element) || {};
        t in r || (r[t] = []),
        r[t].push(n),
        c.set(e.element, r)
    }
    function f(e, t) {
        return (c.get(e.element) || {})[t] || []
    }
    function d(e, t, n) {
        var r = c.get(e.element) || {};
        if (!r[t])
            return !0;
        if (!n)
            return r[t] = [],
            c.set(e.element, r),
            !0;
        var o = r[t].indexOf(n);
        return -1 !== o && r[t].splice(o, 1),
        c.set(e.element, r),
        r[t] && 0 === r[t].length
    }
    var h = ["autopause", "autoplay", "background", "byline", "color", "controls", "dnt", "height", "id", "loop", "maxheight", "maxwidth", "muted", "playsinline", "portrait", "responsive", "speed", "texttrack", "title", "transparent", "url", "width"];
    function v(e) {
        var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
        return h.reduce((function(t, n) {
            var r = e.getAttribute("data-vimeo-".concat(n));
            return (r || "" === r) && (t[n] = "" === r ? 1 : r),
            t
        }
        ), t)
    }
    function m(e, t) {
        var n = e.html;
        if (!t)
            throw new TypeError("An element must be provided");
        if (null !== t.getAttribute("data-vimeo-initialized"))
            return t.querySelector("iframe");
        var r = document.createElement("div");
        return r.innerHTML = n,
        t.appendChild(r.firstChild),
        t.setAttribute("data-vimeo-initialized", "true"),
        t.querySelector("iframe")
    }
    function p(e) {
        var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {}
          , n = arguments.length > 2 ? arguments[2] : void 0;
        return new Promise((function(o, i) {
            if (!r(e))
                throw new TypeError("“".concat(e, "” is not a vimeo.com url."));
            var a = "https://vimeo.com/api/oembed.json?url=".concat(encodeURIComponent(e));
            for (var u in t)
                t.hasOwnProperty(u) && (a += "&".concat(u, "=").concat(encodeURIComponent(t[u])));
            var l = "XDomainRequest"in window ? new XDomainRequest : new XMLHttpRequest;
            l.open("GET", a, !0),
            l.onload = function() {
                if (404 !== l.status)
                    if (403 !== l.status)
                        try {
                            var t = JSON.parse(l.responseText);
                            if (403 === t.domain_status_code)
                                return m(t, n),
                                void i(new Error("“".concat(e, "” is not embeddable.")));
                            o(t)
                        } catch (e) {
                            i(e)
                        }
                    else
                        i(new Error("“".concat(e, "” is not embeddable.")));
                else
                    i(new Error("“".concat(e, "” was not found.")))
            }
            ,
            l.onerror = function() {
                var e = l.status ? " (".concat(l.status, ")") : "";
                i(new Error("There was an error fetching the embed code from Vimeo".concat(e, ".")))
            }
            ,
            l.send()
        }
        ))
    }
    function y(e) {
        if ("string" == typeof e)
            try {
                e = JSON.parse(e)
            } catch (e) {
                return console.warn(e),
                {}
            }
        return e
    }
    function g(e, t, n) {
        if (e.element.contentWindow && e.element.contentWindow.postMessage) {
            var r = {
                method: t
            };
            void 0 !== n && (r.value = n);
            var o = parseFloat(navigator.userAgent.toLowerCase().replace(/^.*msie (\d+).*$/, "$1"));
            o >= 8 && o < 10 && (r = JSON.stringify(r)),
            e.element.contentWindow.postMessage(r, e.origin)
        }
    }
    function w(e, t) {
        var n, r = [];
        if ((t = y(t)).event) {
            if ("error" === t.event)
                f(e, t.data.method).forEach((function(n) {
                    var r = new Error(t.data.message);
                    r.name = t.data.name,
                    n.reject(r),
                    d(e, t.data.method, n)
                }
                ));
            r = f(e, "event:".concat(t.event)),
            n = t.data
        } else if (t.method) {
            var o = function(e, t) {
                var n = f(e, t);
                if (n.length < 1)
                    return !1;
                var r = n.shift();
                return d(e, t, r),
                r
            }(e, t.method);
            o && (r.push(o),
            n = t.value)
        }
        r.forEach((function(t) {
            try {
                if ("function" == typeof t)
                    return void t.call(e, n);
                t.resolve(n)
            } catch (e) {}
        }
        ))
    }
    var b = new WeakMap
      , k = new WeakMap
      , E = {}
      , T = function() {
        function t(e) {
            var n = this
              , i = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
            if (function(e, t) {
                if (!(e instanceof t))
                    throw new TypeError("Cannot call a class as a function")
            }(this, t),
            window.jQuery && e instanceof jQuery && (e.length > 1 && window.console && console.warn && console.warn("A jQuery object with multiple elements was passed, using the first element."),
            e = e[0]),
            "undefined" != typeof document && "string" == typeof e && (e = document.getElementById(e)),
            !function(e) {
                return Boolean(e && 1 === e.nodeType && "nodeName"in e && e.ownerDocument && e.ownerDocument.defaultView)
            }(e))
                throw new TypeError("You must pass either a valid element or a valid id.");
            if ("IFRAME" !== e.nodeName) {
                var a = e.querySelector("iframe");
                a && (e = a)
            }
            if ("IFRAME" === e.nodeName && !r(e.getAttribute("src") || ""))
                throw new Error("The player element passed isn’t a Vimeo embed.");
            if (b.has(e))
                return b.get(e);
            this._window = e.ownerDocument.defaultView,
            this.element = e,
            this.origin = "*";
            var u = new l((function(t, a) {
                if (n._onMessage = function(e) {
                    if (r(e.origin) && n.element.contentWindow === e.source) {
                        "*" === n.origin && (n.origin = e.origin);
                        var o = y(e.data);
                        if (o && "error" === o.event && o.data && "ready" === o.data.method) {
                            var i = new Error(o.data.message);
                            return i.name = o.data.name,
                            void a(i)
                        }
                        var u = o && "ready" === o.event
                          , l = o && "ping" === o.method;
                        if (u || l)
                            return n.element.setAttribute("data-ready", "true"),
                            void t();
                        w(n, o)
                    }
                }
                ,
                n._window.addEventListener("message", n._onMessage),
                "IFRAME" !== n.element.nodeName) {
                    var u = v(e, i);
                    p(o(u), u, e).then((function(t) {
                        var r, o, i, a = m(t, e);
                        return n.element = a,
                        n._originalElement = e,
                        r = e,
                        o = a,
                        i = c.get(r),
                        c.set(o, i),
                        c.delete(r),
                        b.set(n.element, n),
                        t
                    }
                    )).catch(a)
                }
            }
            ));
            if (k.set(this, u),
            b.set(this.element, this),
            "IFRAME" === this.element.nodeName && g(this, "ping"),
            E.isEnabled) {
                var f = function() {
                    return E.exit()
                };
                E.on("fullscreenchange", (function() {
                    E.isFullscreen ? s(n, "event:exitFullscreen", f) : d(n, "event:exitFullscreen", f),
                    n.ready().then((function() {
                        g(n, "fullscreenchange", E.isFullscreen)
                    }
                    ))
                }
                ))
            }
            return this
        }
        var i, a, u;
        return i = t,
        a = [{
            key: "callMethod",
            value: function(e) {
                var t = this
                  , n = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                return new l((function(r, o) {
                    return t.ready().then((function() {
                        s(t, e, {
                            resolve: r,
                            reject: o
                        }),
                        g(t, e, n)
                    }
                    )).catch(o)
                }
                ))
            }
        }, {
            key: "get",
            value: function(e) {
                var t = this;
                return new l((function(r, o) {
                    return e = n(e, "get"),
                    t.ready().then((function() {
                        s(t, e, {
                            resolve: r,
                            reject: o
                        }),
                        g(t, e)
                    }
                    )).catch(o)
                }
                ))
            }
        }, {
            key: "set",
            value: function(e, t) {
                var r = this;
                return new l((function(o, i) {
                    if (e = n(e, "set"),
                    null == t)
                        throw new TypeError("There must be a value to set.");
                    return r.ready().then((function() {
                        s(r, e, {
                            resolve: o,
                            reject: i
                        }),
                        g(r, e, t)
                    }
                    )).catch(i)
                }
                ))
            }
        }, {
            key: "on",
            value: function(e, t) {
                if (!e)
                    throw new TypeError("You must pass an event name.");
                if (!t)
                    throw new TypeError("You must pass a callback function.");
                if ("function" != typeof t)
                    throw new TypeError("The callback must be a function.");
                0 === f(this, "event:".concat(e)).length && this.callMethod("addEventListener", e).catch((function() {}
                )),
                s(this, "event:".concat(e), t)
            }
        }, {
            key: "off",
            value: function(e, t) {
                if (!e)
                    throw new TypeError("You must pass an event name.");
                if (t && "function" != typeof t)
                    throw new TypeError("The callback must be a function.");
                d(this, "event:".concat(e), t) && this.callMethod("removeEventListener", e).catch((function(e) {}
                ))
            }
        }, {
            key: "loadVideo",
            value: function(e) {
                return this.callMethod("loadVideo", e)
            }
        }, {
            key: "ready",
            value: function() {
                var e = k.get(this) || new l((function(e, t) {
                    t(new Error("Unknown player. Probably unloaded."))
                }
                ));
                return l.resolve(e)
            }
        }, {
            key: "addCuePoint",
            value: function(e) {
                var t = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
                return this.callMethod("addCuePoint", {
                    time: e,
                    data: t
                })
            }
        }, {
            key: "removeCuePoint",
            value: function(e) {
                return this.callMethod("removeCuePoint", e)
            }
        }, {
            key: "enableTextTrack",
            value: function(e, t) {
                if (!e)
                    throw new TypeError("You must pass a language.");
                return this.callMethod("enableTextTrack", {
                    language: e,
                    kind: t
                })
            }
        }, {
            key: "disableTextTrack",
            value: function() {
                return this.callMethod("disableTextTrack")
            }
        }, {
            key: "pause",
            value: function() {
                return this.callMethod("pause")
            }
        }, {
            key: "play",
            value: function() {
                return this.callMethod("play")
            }
        }, {
            key: "requestFullscreen",
            value: function() {
                return E.isEnabled ? E.request(this.element) : this.callMethod("requestFullscreen")
            }
        }, {
            key: "exitFullscreen",
            value: function() {
                return E.isEnabled ? E.exit() : this.callMethod("exitFullscreen")
            }
        }, {
            key: "getFullscreen",
            value: function() {
                return E.isEnabled ? l.resolve(E.isFullscreen) : this.get("fullscreen")
            }
        }, {
            key: "unload",
            value: function() {
                return this.callMethod("unload")
            }
        }, {
            key: "destroy",
            value: function() {
                var e = this;
                return new l((function(t) {
                    if (k.delete(e),
                    b.delete(e.element),
                    e._originalElement && (b.delete(e._originalElement),
                    e._originalElement.removeAttribute("data-vimeo-initialized")),
                    e.element && "IFRAME" === e.element.nodeName && e.element.parentNode && e.element.parentNode.removeChild(e.element),
                    e.element && "DIV" === e.element.nodeName && e.element.parentNode) {
                        e.element.removeAttribute("data-vimeo-initialized");
                        var n = e.element.querySelector("iframe");
                        n && n.parentNode && n.parentNode.removeChild(n)
                    }
                    e._window.removeEventListener("message", e._onMessage),
                    t()
                }
                ))
            }
        }, {
            key: "getAutopause",
            value: function() {
                return this.get("autopause")
            }
        }, {
            key: "setAutopause",
            value: function(e) {
                return this.set("autopause", e)
            }
        }, {
            key: "getBuffered",
            value: function() {
                return this.get("buffered")
            }
        }, {
            key: "getCameraProps",
            value: function() {
                return this.get("cameraProps")
            }
        }, {
            key: "setCameraProps",
            value: function(e) {
                return this.set("cameraProps", e)
            }
        }, {
            key: "getChapters",
            value: function() {
                return this.get("chapters")
            }
        }, {
            key: "getCurrentChapter",
            value: function() {
                return this.get("currentChapter")
            }
        }, {
            key: "getColor",
            value: function() {
                return this.get("color")
            }
        }, {
            key: "setColor",
            value: function(e) {
                return this.set("color", e)
            }
        }, {
            key: "getCuePoints",
            value: function() {
                return this.get("cuePoints")
            }
        }, {
            key: "getCurrentTime",
            value: function() {
                return this.get("currentTime")
            }
        }, {
            key: "setCurrentTime",
            value: function(e) {
                return this.set("currentTime", e)
            }
        }, {
            key: "getDuration",
            value: function() {
                return this.get("duration")
            }
        }, {
            key: "getEnded",
            value: function() {
                return this.get("ended")
            }
        }, {
            key: "getLoop",
            value: function() {
                return this.get("loop")
            }
        }, {
            key: "setLoop",
            value: function(e) {
                return this.set("loop", e)
            }
        }, {
            key: "setMuted",
            value: function(e) {
                return this.set("muted", e)
            }
        }, {
            key: "getMuted",
            value: function() {
                return this.get("muted")
            }
        }, {
            key: "getPaused",
            value: function() {
                return this.get("paused")
            }
        }, {
            key: "getPlaybackRate",
            value: function() {
                return this.get("playbackRate")
            }
        }, {
            key: "setPlaybackRate",
            value: function(e) {
                return this.set("playbackRate", e)
            }
        }, {
            key: "getPlayed",
            value: function() {
                return this.get("played")
            }
        }, {
            key: "getQualities",
            value: function() {
                return this.get("qualities")
            }
        }, {
            key: "getQuality",
            value: function() {
                return this.get("quality")
            }
        }, {
            key: "setQuality",
            value: function(e) {
                return this.set("quality", e)
            }
        }, {
            key: "getSeekable",
            value: function() {
                return this.get("seekable")
            }
        }, {
            key: "getSeeking",
            value: function() {
                return this.get("seeking")
            }
        }, {
            key: "getTextTracks",
            value: function() {
                return this.get("textTracks")
            }
        }, {
            key: "getVideoEmbedCode",
            value: function() {
                return this.get("videoEmbedCode")
            }
        }, {
            key: "getVideoId",
            value: function() {
                return this.get("videoId")
            }
        }, {
            key: "getVideoTitle",
            value: function() {
                return this.get("videoTitle")
            }
        }, {
            key: "getVideoWidth",
            value: function() {
                return this.get("videoWidth")
            }
        }, {
            key: "getVideoHeight",
            value: function() {
                return this.get("videoHeight")
            }
        }, {
            key: "getVideoUrl",
            value: function() {
                return this.get("videoUrl")
            }
        }, {
            key: "getVolume",
            value: function() {
                return this.get("volume")
            }
        }, {
            key: "setVolume",
            value: function(e) {
                return this.set("volume", e)
            }
        }],
        a && e(i.prototype, a),
        u && e(i, u),
        t
    }();
    return t || (E = function() {
        var e = function() {
            for (var e, t = [["requestFullscreen", "exitFullscreen", "fullscreenElement", "fullscreenEnabled", "fullscreenchange", "fullscreenerror"], ["webkitRequestFullscreen", "webkitExitFullscreen", "webkitFullscreenElement", "webkitFullscreenEnabled", "webkitfullscreenchange", "webkitfullscreenerror"], ["webkitRequestFullScreen", "webkitCancelFullScreen", "webkitCurrentFullScreenElement", "webkitCancelFullScreen", "webkitfullscreenchange", "webkitfullscreenerror"], ["mozRequestFullScreen", "mozCancelFullScreen", "mozFullScreenElement", "mozFullScreenEnabled", "mozfullscreenchange", "mozfullscreenerror"], ["msRequestFullscreen", "msExitFullscreen", "msFullscreenElement", "msFullscreenEnabled", "MSFullscreenChange", "MSFullscreenError"]], n = 0, r = t.length, o = {}; n < r; n++)
                if ((e = t[n]) && e[1]in document) {
                    for (n = 0; n < e.length; n++)
                        o[t[0][n]] = e[n];
                    return o
                }
            return !1
        }()
          , t = {
            fullscreenchange: e.fullscreenchange,
            fullscreenerror: e.fullscreenerror
        }
          , n = {
            request: function(t) {
                return new Promise((function(r, o) {
                    var i = function e() {
                        n.off("fullscreenchange", e),
                        r()
                    };
                    n.on("fullscreenchange", i);
                    var a = (t = t || document.documentElement)[e.requestFullscreen]();
                    a instanceof Promise && a.then(i).catch(o)
                }
                ))
            },
            exit: function() {
                return new Promise((function(t, r) {
                    if (n.isFullscreen) {
                        var o = function e() {
                            n.off("fullscreenchange", e),
                            t()
                        };
                        n.on("fullscreenchange", o);
                        var i = document[e.exitFullscreen]();
                        i instanceof Promise && i.then(o).catch(r)
                    } else
                        t()
                }
                ))
            },
            on: function(e, n) {
                var r = t[e];
                r && document.addEventListener(r, n)
            },
            off: function(e, n) {
                var r = t[e];
                r && document.removeEventListener(r, n)
            }
        };
        return Object.defineProperties(n, {
            isFullscreen: {
                get: function() {
                    return Boolean(document[e.fullscreenElement])
                }
            },
            element: {
                enumerable: !0,
                get: function() {
                    return document[e.fullscreenElement]
                }
            },
            isEnabled: {
                enumerable: !0,
                get: function() {
                    return Boolean(document[e.fullscreenEnabled])
                }
            }
        }),
        n
    }(),
    function() {
        var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : document
          , t = [].slice.call(e.querySelectorAll("[data-vimeo-id], [data-vimeo-url]"))
          , n = function(e) {
            "console"in window && console.error && console.error("There was an error creating an embed: ".concat(e))
        };
        t.forEach((function(e) {
            try {
                if (null !== e.getAttribute("data-vimeo-defer"))
                    return;
                var t = v(e);
                p(o(t), t, e).then((function(t) {
                    return m(t, e)
                }
                )).catch(n)
            } catch (e) {
                n(e)
            }
        }
        ))
    }(),
    function() {
        var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : document;
        window.VimeoPlayerResizeEmbeds_ || (window.VimeoPlayerResizeEmbeds_ = !0,
        window.addEventListener("message", (function(t) {
            if (r(t.origin) && t.data && "spacechange" === t.data.event)
                for (var n = e.querySelectorAll("iframe"), o = 0; o < n.length; o++)
                    if (n[o].contentWindow === t.source) {
                        n[o].parentElement.style.paddingBottom = "".concat(t.data.data[0].bottom, "px");
                        break
                    }
        }
        )))
    }()),
    T
}
));
