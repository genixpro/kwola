(() => {
  const unique = (values) => [...new Set(values)];
  const functionValue = (value) => typeof value === "function";
  const clean = (value) => String(value ?? "").replace(/\s+/g, " ").trim();
  const handlersFor = (element) => {
    if (!window.kwolaEvents || !window.kwolaEvents.has(element)) return [];
    return unique(window.kwolaEvents.get(element));
  };
  const visible = (bounds) => !(
    bounds.bottom < 0 || bounds.right < 0 ||
    bounds.top > window.innerHeight || bounds.left > window.innerWidth
  );
  const isTextInput = (element) => {
    if (element.tagName === "TEXTAREA" || element.isContentEditable) return true;
    if (element.tagName !== "INPUT") return false;
    const type = (element.getAttribute("type") || "text").toLowerCase();
    return ["", "text", "password", "email"].includes(type);
  };
  const clickTags = new Set(["BUTTON", "A", "AREA", "AUDIO", "VIDEO", "OPTION", "SELECT"]);
  const clickableEvents = new Set([
    "click", "dblclick", "mousedown", "mouseup", "pointerdown", "pointerup",
    "touchend", "touchstart"
  ]);
  const rightClickEvents = new Set(["contextmenu", "auxclick", "mousedown", "mouseup"]);
  const typeEvents = new Set(["keydown", "keypress", "keyup"]);
  const targets = [];
  try {
    for (const element of document.querySelectorAll("*")) {
      const bounds = element.getBoundingClientRect();
      if (!visible(bounds)) continue;
      const style = window.getComputedStyle(element);
      const padding = {
        left: Number.parseFloat(style.paddingLeft) || 0,
        right: Number.parseFloat(style.paddingRight) || 0,
        top: Number.parseFloat(style.paddingTop) || 0,
        bottom: Number.parseFloat(style.paddingBottom) || 0,
      };
      const width = bounds.width - padding.left - padding.right - 4;
      const height = bounds.height - padding.top - padding.bottom - 4;
      if (width < 1 || height < 1) continue;
      const events = handlersFor(element);
      const fullScreen = width > window.innerWidth * 0.8 && height > window.innerHeight * 0.8;
      const nativeClick = clickTags.has(element.tagName) ||
        (element.tagName === "INPUT" && !isTextInput(element));
      const inlineClick = !fullScreen && [
        element.onclick, element.onmousedown, element.onmouseup, element.onpointerdown,
        element.onpointerup, element.ontouchend, element.ontouchstart
      ].some(functionValue);
      const inlineRightClick = !fullScreen &&
        [element.oncontextmenu, element.onauxclick].some(functionValue);
      const inlineType = !fullScreen &&
        [element.onkeydown, element.onkeypress, element.onkeyup].some(functionValue);
      const scrollable = (
        style.overflowY === "scroll" || style.overflowY === "auto" ||
        (style.overflowY === "visible" && ["HTML", "BODY"].includes(element.tagName))
      ) && element.scrollHeight > element.clientHeight;
      const center = document.elementFromPoint(
        bounds.left + bounds.width / 2,
        bounds.top + bounds.height / 2
      );
      const onTop = center === null || element.contains(center) || center.contains(element);
      const keywordValues = [
        element.innerText, element.className, element.getAttribute("name"),
        element.id, element.getAttribute("type"), element.getAttribute("placeholder"),
        element.getAttribute("title"), element.getAttribute("aria-label"),
        element.getAttribute("aria-placeholder"), element.getAttribute("aria-roledescription")
      ];
      targets.push({
        left: Math.round(bounds.left + padding.left + 2),
        right: Math.round(bounds.right - padding.right - 2),
        top: Math.round(bounds.top + padding.top + 2),
        bottom: Math.round(bounds.bottom - padding.bottom - 2),
        elementType: element.tagName.toLowerCase(),
        keywords: clean(keywordValues.join(" ")).toLowerCase().slice(0, 1024),
        canClick: nativeClick || inlineClick || events.some((event) => clickableEvents.has(event)),
        canRightClick: inlineRightClick || events.some((event) => rightClickEvents.has(event)),
        canType: isTextInput(element) || inlineType || events.some((event) => typeEvents.has(event)),
        canScroll: scrollable,
        canScrollUp: scrollable && element.scrollTop > 5,
        canScrollDown: scrollable && element.scrollHeight - element.scrollTop - element.clientHeight >= 5,
        isOnTop: onTop,
        attributes: {
          href: clean(element.getAttribute("href")),
          src: clean(element.getAttribute("src")),
          id: clean(element.id),
          name: clean(element.getAttribute("name")),
          type: clean(element.getAttribute("type")),
        },
      });
    }
    return {targets, width: window.innerWidth, height: window.innerHeight, error: null};
  } catch (error) {
    return {targets, width: window.innerWidth, height: window.innerHeight, error: String(error)};
  }
})()
