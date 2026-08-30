const babel = require("@babel/core");
const kwola = require("babel-plugin-kwola").default;
const readline = require("readline");

process.env.KWOLA_ENABLE_LINE_COUNTING = "true";
process.env.KWOLA_ENABLE_EVENT_HANDLER_TRACKING = "true";

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

input.on("line", (line) => {
  let request;
  try {
    request = JSON.parse(line);
    const seed = Number.parseInt(request.resourceId.slice(0, 8), 16) / 0xffffffff;
    const previousRandom = Math.random;
    Math.random = () => seed;
    const result = babel.transformSync(
      Buffer.from(request.source, "base64").toString("utf8"),
      {
        filename: request.resourceId,
        plugins: [[kwola, {}, request.resourceId]],
        retainLines: true,
        sourceType: request.sourceType,
      },
    );
    Math.random = previousRandom;
    const code = Buffer.from(result.code, "utf8").toString("base64");
    process.stdout.write(`${JSON.stringify({ ok: true, code })}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ ok: false, error: String(error) })}\n`);
  }
});
