(function() {
    function E(b) {
        var a = {};
        decodeURIComponent(b).split("&").forEach(function(b) {
            b = b.split("=");
            a[b[0]] = (b[1] || "").replace(/\+/g, " ")
        });
        return a
    }
    function m(b) {
        var a = 0
          , c = document.querySelector("#smart-cart")
          , e = document.querySelector("#smart-cart-qty")
          , g = {};
        minicartPaypal.minicart.cart.subtotal(g);
        g = g.currency;
        minicartPaypal.minicart.cart.items().forEach(function(b) {
            a += b._data.quantity
        });
        1 == b && setTimeout(function() {
            c.smScale(400, 18);
            e.smColorSwap(400);
            setTimeout(function() {
                c.smScale(400, 12);
                e.smColorSwap(400);
                setTimeout(function() {
                    c.smScale(400, 18);
                    e.smColorSwap(400);
                    setTimeout(function() {
                        c.smScale(400, 12);
                        e.smColorSwap(400)
                    }, 400)
                }, 400)
            }, 400)
        }, 400);
        0 < a ? (c.smFadeTo(200, 1),
        e.innerText = a ? a : "") : (e.innerText = a ? a : 0,
        c.smFadeTo(200, 0));
        F()
    }
    function y() {
        var b;
        b = window.innerWidth <= c.site_width ? c.side_offset : (window.innerWidth - c.site_width) / 2 + c.side_offset;
        switch (v) {
        case 0:
            d.style.left = b + "px";
            d.style.right = "initial";
            break;
        case 1:
            d.style.left = "initial",
            d.style.right = b + "px"
        }
    }
    function F() {
        var b = document.querySelector("#paypal-button-container");
        if (window.paypal)
            n || (n = setInterval(function() {
                window.paypal && (b.innerHTML = "",
                [paypal.FUNDING.PAYPAL, paypal.FUNDING.CREDIT, paypal.FUNDING.CARD].forEach(function(a) {
                    a = paypal.Buttons({
                        style: {
                            shape: "pill"
                        },
                        fundingSource: a,
                        createOrder: function(a, b) {
                            var c = minicartPaypal.minicart.cart._items.map(function(a) {
                                var b = {};
                                b.name = a._data.item_name;
                                b.unit_amount = {
                                    currency_code: minicartPaypal.minicart.cart._settings.currency_code,
                                    value: parseFloat(a._data.amount).toFixed(2)
                                };
                                b.quantity = a._data.quantity;
                                return b
                            });
                            c.forEach(function(a) {});
                            return b.order.create({
                                purchase_units: [{
                                    amount: {
                                        currency_code: minicartPaypal.minicart.cart._settings.currency_code,
                                        value: parseFloat(minicartPaypal.minicart.cart.total()).toFixed(2),
                                        breakdown: {
                                            item_total: {
                                                currency_code: minicartPaypal.minicart.cart._settings.currency_code,
                                                value: parseFloat(minicartPaypal.minicart.cart.total()).toFixed(2)
                                            }
                                        }
                                    },
                                    items: c
                                }]
                            })
                        },
                        onClick: function() {
                            setTimeout(function() {
                                var a = document.querySelector("#paypal-button-container iframe");
                                a && a.scrollIntoView()
                            }, 2E3)
                        },
                        onApprove: function(a, b) {
                            return b.order.capture().then(function(a) {
                                minicartPaypal.minicart.cart.destroy();
                                m();
                                a = document.createElement("div");
                                a.classList.add("modal");
                                a.innerHTML = '\n\t\t\t\t\t\t\t\t\t<div id="approve-window" class="modal-dialog display-7" style="display:flex;height:auto;">\n\t\t\t\t\t\t\t\t\t\t<div class="modal-content" style="height:auto;">\n\t\t\t\t\t\t\t\t\t\t\t<div class="modal-header" style="border:none;">\n\t\t\t\t\t\t\t\t\t\t\t\t<button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>\n\t\t\t\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t\t\t\t\t<div class="modal-body align-center">\n\t\t\t\t\t\t\t\t\t\t\t\t<h6 class="display-5">Thank your for your purchase</h6>\n\t\t\t\t\t\t\t\t\t\t\t\t<div class="btn btn-primary" data-bs-dismiss="modal">Close</div>\n\t\t\t\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t\t\t\t\t<div class="modal-footer" style="border:none;display:block;">\n\t\t\t\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t\t\t</div>';
                                (new bootstrap.Modal(a)).show()
                            })
                        }
                    });
                    a.isEligible() && a.render("#paypal-button-container")
                }),
                clearInterval(n))
            }, 1E3));
        else if (z) {
            b.querySelector(".dummy-template-paypal").remove();
            var a = document.createElement("p");
            a.id = "stripe-pay";
            a.style = "display:flex;justify-content:center;align-items:center;background:#2c2e2f;color:#fff;text-align:center;border-radius:23px;width:100%;font-size:1.5em;cursor:pointer; margin-top:1rem; fill:#fff;height:55px;";
            a.innerText = c.stripe_button;
            document.querySelector("#paypal-button-container").append(a);
            a.addEventListener("click", function() {
                var a, b = minicartPaypal.minicart.cart.items().map(function(b) {
                    a || (a = b._data.account);
                    return {
                        amount: b._data.amount,
                        item_name: b._data.item_name,
                        quantity: b._data.quantity,
                        currency_code: minicartPaypal.minicart.cart._settings.currency_code,
                        sign: b._data.sign
                    }
                });
                return G({
                    account: a,
                    success_url: /^https?:/.test(c.returnURL) ? c.returnURL : void 0,
                    cancel_url: /^https?:/.test(c.cancel_returnURL) ? c.cancel_returnURL : void 0,
                    items: b
                }, function(a, b) {
                    if (a) {
                        var c = document.querySelector("#paypal-button-container");
                        c.innerHTML = "";
                        var h = document.createElement("div");
                        h.innerText = "Error: " + a.message || a;
                        h.classList.add("dummy-template-paypal");
                        c.append(h)
                    } else
                        document.location.href = b.url
                })
            })
        } else if (A.length) {
            if ((a = b.parentElement.querySelector(".dummy-template-paypal")) && a.remove(),
            !document.querySelector("#cash-on-delivery")) {
                var h = document.createElement("p");
                document.querySelector("#paypal-button-container").append(h);
                h.id = "cash-on-delivery";
                h.style = "z-index:3000;display:flex;justify-content:center;align-items:center;background:#128C7E;color:#fff;text-align:center;border-radius:23px;width:100%;font-size:1.5em;cursor:pointer; margin-top:1rem; fill:#fff;height:55px;";
                h.innerHTML = c.whatsapp_button;
                h.addEventListener("click", function(a) {
                    a.stopPropagation();
                    B();
                    h.setAttribute("disabled", !0);
                    minicartPaypal.minicart.cart && (minicartPaypal.minicart.cart.destroy(),
                    m());
                    setTimeout(function() {
                        h.removeAttribute("disabled")
                    }, 1E3)
                })
            }
        } else
            clearInterval(n),
            document.querySelector("#paypal-button-container .dummy-template-paypal").innerText = "Error: Invalid PayPal Client ID."
    }
    function G(b, a) {
        var c = new XMLHttpRequest;
        c.open("POST", "https://p.electricblaze.com/stripe/checkout");
        c.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        c.onload = function() {
            var b;
            try {
                b = JSON.parse(c.response)
            } catch (d) {
                return a(Error("Invalid server response"))
            }
            if (200 > c.status || 400 <= c.status)
                return a(Error(b.message));
            a(null, b)
        }
        ;
        c.onerror = function() {
            a(Error("Failed to load content"))
        }
        ;
        c.send(JSON.stringify(b))
    }
    function H() {
        if (!C && document.body) {
            switch (c.shopcart_position) {
            case "left":
                v = 0;
                break;
            case "right":
                v = 1;
                break;
            default:
                console.log("Shopcart position (orientation) is wrong");
                return
            }
            minicartPaypal.minicart.render({
                strings: {
                    button: c.buy_now_title,
                    cart_title: c.cart_title,
                    total: c.total
                }
            });
            window.location.href == c.returnURL && minicartPaypal.minicart.reset();
            var b = document.createElement("div");
            b.innerHTML = c.shopcartHtml;
            document.querySelector("body").append(b);
            b = document.createElement("style");
            b.innerHTML = "#giftcard-to-smartcart {font-size: " + c.gift_icon_size + "px; color: " + c.gift_icon_color + "; background-color: " + c.gift_back_color + "}";
            document.querySelector("body").append(b);
            window.addEventListener("resize", function() {
                y()
            });
            m();
            document.querySelector("#smart-cart").addEventListener("click", function(a) {
                a.stopPropagation();
                minicartPaypal.minicart.view.show();
                minicartPaypal.minicart.view.redraw()
            });
            document.addEventListener("click", function(a) {
                var b = a.target.getAttribute("href");
                if (b && (I.test(b) || J.test(b))) {
                    var e = new URL(b)
                      , g = e.searchParams.get("cmdsm") || "";
                    if ("_cart" === g)
                        a.stopPropagation(),
                        a.preventDefault(),
                        w = !0,
                        a = E(b.substring(b.indexOf("?") + 1)),
                        g = {
                            amount: a.amount,
                            bn: a.bn,
                            business: a.business,
                            currency_code: a.currency_code,
                            item_name: a.item_name,
                            item_number: a.item_number,
                            account: a.account,
                            sign: a.sign,
                            shipping: a.shipping,
                            shipping2: a.shipping2,
                            "return": c.returnURL,
                            cancel_return: c.cancel_returnURL,
                            notifyURL: c.notifyURL
                        },
                        a.nopayment_whatsapp && (g.nopayment_whatsapp = a.nopayment_whatsapp),
                        a.nopayment_email && (g.nopayment_email = a.nopayment_email),
                        !1 !== minicartPaypal.minicart.cart.add(g) && d && (d.smFadeTo(200, 1),
                        m(!0)),
                        w = !1;
                    else if ("_xclick" === g) {
                        var k = function() {
                            if (!l)
                                if (window.paypal)
                                    l = setInterval(function() {
                                        r.innerHTML = "";
                                        [paypal.FUNDING.PAYPAL, paypal.FUNDING.CREDIT, paypal.FUNDING.CARD].forEach(function(a) {
                                            a = paypal.Buttons({
                                                style: {
                                                    shape: "pill"
                                                },
                                                fundingSource: a,
                                                createOrder: function(a, b) {
                                                    return b.order.create({
                                                        purchase_units: [{
                                                            style: {
                                                                layout: "horizontal"
                                                            },
                                                            amount: {
                                                                currency_code: p,
                                                                value: q * f.quantity,
                                                                breakdown: {
                                                                    item_total: {
                                                                        currency_code: p,
                                                                        value: q * f.quantity
                                                                    }
                                                                }
                                                            },
                                                            items: [{
                                                                name: f.name || "",
                                                                unit_amount: {
                                                                    currency_code: p,
                                                                    value: q
                                                                },
                                                                quantity: f.quantity
                                                            }]
                                                        }]
                                                    })
                                                },
                                                onApprove: function(a, b) {
                                                    return b.order.capture().then(function(a) {
                                                        minicartPaypal.minicart.cart.destroy();
                                                        m();
                                                        a = document.createElement("div");
                                                        a.classList.add("modal");
                                                        a.innerHTML = '\n\t\t\t\t\t\t\t\t\t\t\t<div id="approve-window" class="modal-dialog display-7" style="display:flex;height:auto;">\n\t\t\t\t\t\t\t\t\t\t\t\t<div class="modal-content" style="height:auto;">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<div class="modal-header" style="border:none;">\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t<button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>\n\t\t\t\t\t\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t\t\t\t\t\t\t<div class="modal-body align-center">\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t<h6 class="display-5">Thank you for your order!</h6>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t<div class="btn btn-primary" data-bs-dismiss="modal">Close</div>\n\t\t\t\t\t\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t\t\t\t\t\t\t<div class="modal-footer" style="border:none;display:block;">\n\t\t\t\t\t\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t\t\t\t\t</div>';
                                                        (new bootstrap.Modal(a)).show()
                                                    })
                                                }
                                            });
                                            a.isEligible() && a.render(r)
                                        });
                                        clearInterval(l)
                                    }, 1E3);
                                else if (z) {
                                    r.querySelector(".dummy-template-paypal").remove();
                                    var a = document.createElement("p");
                                    a.id = "stripe-buynow-pay";
                                    a.style = "z-index:3000;display:flex;justify-content:center;align-items:center;background:#2c2e2f;color:#fff;text-align:center;border-radius:23px;width:100%;font-size:1.5em;cursor:pointer; margin-top:1rem; fill:#fff;height:55px;";
                                    a.innerHTML = c.stripe_button;
                                    a.addEventListener("click", function(a) {
                                        a.stopPropagation();
                                        a = b + "&quantity=" + t.value;
                                        /^https?:/.test(c.returnURL) && (a = a + "&success_url=" + encodeURIComponent(c.returnURL));
                                        /^https?:/.test(c.cancel_returnURL) && (a = a + "&cancel_url=" + encodeURIComponent(c.cancel_returnURL));
                                        document.location.href = a
                                    });
                                    r.append(a)
                                } else if (A.length) {
                                    r.querySelector(".dummy-template-paypal").remove();
                                    var d = document.createElement("p");
                                    d.id = "cash-on-delivery";
                                    d.style = "z-index:3000;display:flex;justify-content:center;align-items:center;background:#128C7E;color:#fff;text-align:center;border-radius:23px;width:100%;font-size:1.5em;cursor:pointer; margin-top:1rem; fill:#fff;height:55px;";
                                    d.innerHTML = c.whatsapp_button || "Place order";
                                    d.addEventListener("click", function(a) {
                                        a.stopPropagation();
                                        a.preventDefault();
                                        a = {
                                            amount: q,
                                            item_name: n,
                                            quantity: f.quantity,
                                            currency_code: p,
                                            nopayment_whatsapp: e.searchParams.get("nopayment_whatsapp")
                                        };
                                        B([a]);
                                        d.setAttribute("disabled", !0);
                                        setTimeout(function() {
                                            d.removeAttribute("disabled")
                                        }, 1E3)
                                    });
                                    r.append(d)
                                } else
                                    clearInterval(l),
                                    r.querySelector(".dummy-template-paypal").innerText = "Error: Invalid PayPal Client ID."
                        };
                        a.preventDefault();
                        a.stopPropagation();
                        a = document.createElement("div");
                        a.classList.add("modal");
                        var q = e.searchParams.get("amount") || ""
                          , p = e.searchParams.get("currency_code") || ""
                          , n = e.searchParams.get("item_name") || ""
                          , f = {};
                        f.name = n;
                        f.quantity = 1;
                        a.innerHTML = '\n\t\t\t\t<div id="buy-now-window" class="modal-dialog display-7" style="display:flex;height:auto;">\n\t\t\t\t\t<div class="modal-content" style="height:auto;">\n\t\t\t\t\t\t<div class="modal-header" style="border:none;">\n\t\t\t\t\t\t\t<h6 class="modal-title" id="staticBackdropLabel"><b>' + c.buy_now_title + '</b></h6>\n\t\t\t\t\t\t\t<button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>\n\t\t\t\t\t\t</div>\n\t\t\t\t\t\t<div class="modal-body">\n\t\t\t\t\t\t\t<div class="row my-3">\n\t\t\t\t\t\t\t\t<div class="col">\n\t\t\t\t\t\t\t\t\t<div class="item-name">' + (n || "Item Name not found") + '</div>\n\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t\t<div class="minicart-details-quantity col-auto">\n\t\t\t\t\t\t\t\t\t<span class="smartcart-minus">&minus;</span>\n\t\t\t\t\t\t\t\t\t<input type="number" class="smartcart-quantity" value="' + f.quantity + '" autocomplete="off"/>\n\t\t\t\t\t\t\t\t\t<span class="smartcart-plus">&plus;</span>\n\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t<div class="row align-items-center">\n\t\t\t\t\t\t\t\t<div class="col">\n\t\t\t\t\t\t\t\t\t<div class="item-price align-right">' + (q + " " + p) + '</div>\n\t\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t</div>\n\t\t\t\t\t\t<div class="modal-footer" style="border:none;display:block;">\n\t\t\t\t\t\t\t<div id="paypal-button-buynow">\n\t\t\t\t\t\t\t\t<div class="dummy-template-paypal">PAYPAL LOADING</div>\n\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t</div>\n\t\t\t\t\t</div>\n\t\t\t\t</div>';
                        var u = new bootstrap.Modal(a)
                          , r = a.querySelector("#paypal-button-buynow")
                          , l = void 0;
                        a.addEventListener("shown.bs.modal", function(a) {
                            k()
                        });
                        var g = a.querySelector(".smartcart-minus")
                          , t = a.querySelector(".smartcart-quantity")
                          , v = a.querySelector(".smartcart-plus")
                          , x = a.querySelector(".item-price");
                        g.addEventListener("click", function(a) {
                            1 >= parseInt(f.quantity) || (f.quantity = parseInt(f.quantity) - 1,
                            t.value = f.quantity,
                            x.innerText = parseFloat(t.value * q).toFixed(2) + " " + p,
                            k())
                        });
                        v.addEventListener("click", function(a) {
                            f.quantity = parseInt(f.quantity) + 1;
                            t.value = f.quantity;
                            x.innerText = parseFloat(t.value * q).toFixed(2) + " " + p;
                            k()
                        });
                        t.addEventListener("change", function(a) {
                            f.quantity = parseInt(a.target.value);
                            t.value = f.quantity;
                            x.innerText = parseFloat(a.target.value * q).toFixed(2) + " " + p;
                            k()
                        });
                        a.addEventListener("hidden.bs.modal", function(a) {
                            u.dispose()
                        });
                        u.show()
                    }
                }
            });
            minicartPaypal.minicart.cart.on("change", function(a) {
                w || m()
            });
            minicartPaypal.minicart.cart.on("remove", function() {
                m()
            });
            d = document.querySelector("#smart-cart");
            d.style.top = c.shopcart_top_offset + "px";
            d.style.fontSize = c.shopcart_icon_size + "px";
            d.style.width = c.shopcart_icon_size + 18 + "px";
            d.style.height = c.shopcart_icon_size + 18 + "px";
            d.style.color = c.shopcart_icon_color;
            d.style.backgroundColor = c.shopcart_back_color;
            d.querySelector("span").style.fontSize = c.sc_count_size + "px";
            d.querySelector("span").style.color = c.sc_count_color;
            d.querySelector("span").style.backgroundColor = c.sc_count_back_color;
            b = {};
            minicartPaypal.minicart.cart.subtotal(b);
            b = b.currency;
            shopcartWidth = d.offsetWidth;
            y();
            C = !0
        }
    }
    function u(b) {
        for (var a in b)
            D.hasOwnProperty(a) && (c[a] = b[a]);
        H();
        -1 !== location.href.search(/\?success/) && (minicartPaypal.minicart.cart && (minicartPaypal.minicart.cart.destroy(),
        m()),
        b = document.createElement("div"),
        b.classList.add("modal"),
        b.innerHTML = '\n\t\t\t\t\t<div id="approve-window" class="modal-dialog display-7" style="display:flex;height:auto;">\n\t\t\t\t\t\t<div class="modal-content" style="height:auto;">\n\t\t\t\t\t\t\t<div class="modal-header" style="border:none;">\n\t\t\t\t\t\t\t\t<button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>\n\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t<div class="modal-body align-center">\n\t\t\t\t\t\t\t\t<h6 class="display-5">Thank you for your order!</h6>\n\t\t\t\t\t\t\t\t<div data-bs-dismiss="modal" class="btn btn-primary">Close</div>\n\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t\t<div class="modal-footer" style="border:none;display:block;">\n\t\t\t\t\t\t\t</div>\n\t\t\t\t\t\t</div>\n\t\t\t\t\t</div>',
        (new bootstrap.Modal(b)).show())
    }
    var D = {
        shopcart_position: "right",
        site_width: 1150,
        side_offset: 20,
        shopcart_top_offset: 150,
        gift_icon_color: "#14142b",
        gift_back_color: "#fedb01",
        gift_icon_size: 15,
        shopcart_icon_color: "#ffffff",
        shopcart_back_color: "#14142b",
        shopcart_icon_size: 50,
        sc_count_color: "#14142b",
        sc_count_back_color: "#fedb01",
        sc_count_size: 16,
        returnURL: window.location.origin + window.location.pathname + "?payment_success=true",
        cancel_returnURL: window.location.origin + window.location.pathname + "?failure",
        shopcartCSSLink: '<link rel="stylesheet" href="smart-cart/minicart-theme.css" type="text/css">',
        giftCardHtml: '<i id="giftcard-to-smartcart" class="shoppingcart-icons">&#xe308;</i>',
        shopcartHtml: '<i id="smart-cart" role="button" class="shoppingcart-icons display-7" data-placement="left"><svg fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M21 19H5.2l-3-15H0V2h3.8l.8 4H23l-1.2 9H6.4l.4 2H21v2zM6 13h14l.7-5H5l1 5zm2 7c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm10 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg><span id="smart-cart-qty"></span></div></i>',
        cart_title: "Your Cart",
        buy_now_title: "Buy Now",
        total: "Total",
        stripe_button: "Pay",
        whatsapp_button: "Order on WhatsApp",
        whatsapp_message: "Hi, I\u2019d like to order the following:"
    }, I = /https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)/, J = /mbrext\:\/\//, c = D, C = !1, w = !1, d, v;
    Element.prototype.smFadeTo = function(b, a) {
        function c() {
            d.style.display = "none"
        }
        var d = this;
        d.style.transitionDuration = b + "ms";
        d.style.opacity = a;
        0 === a ? d.addEventListener("transitionend", c(), {
            once: !0
        }) : (d.removeEventListener("transitionend", c()),
        d.style.display = "flex")
    }
    ;
    Element.prototype.smScale = function(b, a) {
        this.style.transitionDuration = b + "ms";
        this.style.transitionTimingFunction = "ease-out";
        this.style.padding = a + "px"
    }
    ;
    Element.prototype.smColorSwap = function(b) {
        this.style.transitionDuration = b + "ms";
        b = this.style.color;
        this.style.color = this.style.backgroundColor;
        this.style.backgroundColor = b
    }
    ;
    var l = document.createElement("div");
    l.classList.add("payment-notification-container");
    l.style.position = "absolute";
    l.style.top = "0";
    l.style.right = "0";
    var z = document.querySelector("a[href*=account]")
      , A = document.querySelectorAll('a[href*="mbrext://smart-cart"]')
      , n = void 0
      , B = function(b, a) {
        var d = ""
          , e = "";
        b = b || minicartPaypal.minicart.cart.items().map(function(a) {
            e = a.currency_code ? a.currency_code.toUpperCase() : "";
            d = "+" + a._data.nopayment_whatsapp.replace(/\s+/g, "");
            return {
                amount: a._data.amount,
                item_name: a._data.item_name,
                quantity: a._data.quantity,
                currency_code: minicartPaypal.minicart.cart._settings.currency_code,
                sign: a._data.sign
            }
        });
        !d && b.length && (d = "+" + b[0].nopayment_whatsapp.trim());
        var g = 0;
        b.length && (e = b[0].currency_code.toUpperCase());
        var k = c.whatsapp_message + "\n";
        b.forEach(function(a, b) {
            g += a.amount * a.quantity;
            k = k + a.item_name + " x" + a.quantity + " " + e + " " + a.amount * a.quantity + "\n"
        });
        k = k + c.total + " " + e + " " + g;
        window.open("https://wa.me/" + d + "?text=" + encodeURIComponent(k))
    };
    if ("undefined" === typeof minicartPaypal || "undefined" === typeof minicartPaypal.minicart)
        return -1;
    window.smartCartDefSettings && u(window.smartCartDefSettings);
    "function" === typeof define && define.amd ? define(function() {
        return u
    }) : "object" == typeof exports ? module.exports = u : window.mcSmartCart = u
}
)();
mcSmartCart("undefined" === typeof smartCartDefSettings ? null : smartCartDefSettings);
