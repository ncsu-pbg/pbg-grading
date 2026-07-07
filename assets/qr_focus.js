/* QR box focus manager — makes the handheld reader "just work".
 *
 * Why: the reader emulates a keyboard, so its scan lands in whichever element
 * has focus. The QR input's HTML autofocus became unreliable once it moved
 * inside a Bootstrap tab pane, so after load the focus could be nowhere and a
 * scan would type into the void.
 *
 * What this does:
 *  1. Focuses the QR box once the Dash layout has rendered.
 *  2. Re-focuses it when the QR Scan tab is clicked.
 *  3. Scan-anywhere: if printable keystrokes arrive while no input/textarea
 *     is focused and the QR box is visible, focus it so the scan lands there
 *     (the triggering character is preserved by focusing during keydown).
 */
(function () {
    function qrBox() { return document.getElementById("qr-scan"); }

    function qrVisible() {
        var el = qrBox();
        return el && !el.disabled && el.offsetParent !== null; // hidden tab -> null
    }

    function focusQR() {
        if (qrVisible()) { qrBox().focus(); }
    }

    // 1. After load — Dash renders asynchronously, so retry briefly.
    var tries = 0;
    var timer = setInterval(function () {
        if (qrBox()) { focusQR(); clearInterval(timer); }
        else if (++tries > 40) { clearInterval(timer); }   // give up after ~10 s
    }, 250);

    // 2. Re-focus when the QR Scan tab is selected (tab pane shows async).
    document.addEventListener("click", function (e) {
        var t = e.target.closest ? e.target.closest(".nav-link") : null;
        if (t && /qr/i.test(t.textContent || "")) { setTimeout(focusQR, 100); }
    }, true);

    // 3. Scan-anywhere: stray printable keystrokes route to the QR box.
    document.addEventListener("keydown", function (e) {
        if (e.ctrlKey || e.metaKey || e.altKey) { return; }
        if (!e.key || e.key.length !== 1) { return; }       // printable only
        var a = document.activeElement;
        var typing = a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" ||
                           a.isContentEditable);
        if (!typing) { focusQR(); }   // focus during keydown keeps this char
    }, true);
})();
