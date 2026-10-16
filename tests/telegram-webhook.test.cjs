const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const ts = require("typescript");

const source = fs.readFileSync("supabase/functions/telegram-webhook/index.ts", "utf8");

function extract(start, end) {
  const a = source.indexOf(start);
  const b = source.indexOf(end, a + start.length);
  assert.ok(a >= 0 && b > a, `Cannot extract ${start}`);
  return source.slice(a, b);
}

function compile(snippet, context) {
  const js = ts.transpileModule(snippet, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
  }).outputText;
  return vm.runInNewContext(js + "\n({ findNextSentCandidate: typeof findNextSentCandidate === 'function' ? findNextSentCandidate : undefined, answerCallbackQuery: typeof answerCallbackQuery === 'function' ? answerCallbackQuery : undefined })", context);
}

function fakeSupabase(rows) {
  const calls = [];
  return {
    calls,
    from(table) {
      assert.equal(table, "editor_queue");
      const filters = [];
      let direction = true;
      const query = {
        select() { return query; },
        eq(key, value) { filters.push(row => row[key] === value); return query; },
        neq(key, value) { filters.push(row => row[key] !== value); return query; },
        gt(key, value) { filters.push(row => row[key] > value); return query; },
        lt(key, value) { filters.push(row => row[key] < value); return query; },
        order(key, options) { if (key === "telegram_message_id") direction = options.ascending; return query; },
        limit() { return query; },
        async maybeSingle() {
          calls.push(filters.length);
          const matches = rows.filter(row => filters.every(f => f(row)))
            .sort((a, b) => direction ? a.telegram_message_id - b.telegram_message_id : b.telegram_message_id - a.telegram_message_id);
          return { data: matches[0] ?? null, error: null };
        },
      };
      return query;
    },
  };
}

const findSource = extract("async function findNextSentCandidate(", "async function findPendingCandidate(");
const ackSource = extract("async function answerCallbackQuery(", "async function editMessageReplyMarkup(");

test("next candidate excludes the current article when it is the only sent card", async () => {
  const db = fakeSupabase([{ id: 2016, status: "sent", telegram_chat_id: 5, telegram_message_id: 5484 }]);
  const { findNextSentCandidate } = compile(findSource, { supabase: db });
  assert.equal(await findNextSentCandidate(5, 5484, 2016), null);
});

test("next candidate chooses the nearest later different article", async () => {
  const db = fakeSupabase([
    { id: 2016, status: "sent", telegram_chat_id: 5, telegram_message_id: 5484 },
    { id: 2017, status: "sent", telegram_chat_id: 5, telegram_message_id: 5485 },
    { id: 2018, status: "sent", telegram_chat_id: 5, telegram_message_id: 5490 },
  ]);
  const { findNextSentCandidate } = compile(findSource, { supabase: db });
  assert.equal((await findNextSentCandidate(5, 5484, 2016)).id, 2017);
});

test("next candidate wraps to an earlier different article", async () => {
  const db = fakeSupabase([
    { id: 2015, status: "sent", telegram_chat_id: 5, telegram_message_id: 5480 },
    { id: 2016, status: "sent", telegram_chat_id: 5, telegram_message_id: 5484 },
  ]);
  const { findNextSentCandidate } = compile(findSource, { supabase: db });
  assert.equal((await findNextSentCandidate(5, 5484, 2016)).id, 2015);
});

test("navigation never selects cards from another chat or completed cards", async () => {
  const db = fakeSupabase([
    { id: 1, status: "sent", telegram_chat_id: 6, telegram_message_id: 5485 },
    { id: 2, status: "approved", telegram_chat_id: 5, telegram_message_id: 5486 },
  ]);
  const { findNextSentCandidate } = compile(findSource, { supabase: db });
  assert.equal(await findNextSentCandidate(5, 5484, 2016), null);
});

test("expired Telegram callback acknowledgement is nonfatal", async () => {
  const { answerCallbackQuery } = compile(ackSource, {
    telegram: async () => { throw new Error("Bad Request: query is too old and response timeout expired or query ID is invalid"); },
    console: { warn() {} },
  });
  await assert.doesNotReject(answerCallbackQuery("expired"));
});

test("unexpected Telegram acknowledgement failures still propagate", async () => {
  const { answerCallbackQuery } = compile(ackSource, {
    telegram: async () => { throw new Error("Telegram API unavailable"); },
    console: { warn() {} },
  });
  await assert.rejects(answerCallbackQuery("x"), /Telegram API unavailable/);
});

test("next button uses clicked message as cursor and excludes current queue item", () => {
  const nextHandler = extract("async function handleNextCandidate(", "async function handleNextCommand(");
  assert.match(nextHandler, /const cursorMessageId = messageId;/);
  assert.match(nextHandler, /sendNextCandidate\(chatId, cursorMessageId, queue\.id\)/);
});
