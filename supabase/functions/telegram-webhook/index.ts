import { createClient } from "https://esm.sh/@supabase/supabase-js@2.115.0";

const BOT_TOKEN = requireEnv("TELEGRAM_BOT_TOKEN");
const CHANNEL_ID = requireEnv("TELEGRAM_CHANNEL_ID");
const SUPABASE_URL = requireEnv("SUPABASE_URL");
const SUPABASE_SERVICE_ROLE_KEY = requireEnv("SUPABASE_SERVICE_ROLE_KEY");

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

type FeedbackType = "text_correction" | "topic_mismatch" | "other_rejection";

type QueueRow = {
  id: number;
  article_id: number;
  status: string;
  revision: number;
  telegram_chat_id: number | null;
  telegram_message_id: number | null;
};

type CallbackAction =
  | { kind: "publish"; queueId: number; revision?: number }
  | { kind: "reject"; queueId: number; revision?: number }
  | { kind: "back"; queueId: number; revision?: number }
  | { kind: "feedback"; feedbackType: FeedbackType; queueId: number; revision: number }
  | { kind: "prefilter"; action: "details" | "activate" | "reject"; proposalId: number }
  | { kind: "done" };

function requireEnv(name: string): string {
  const value = Deno.env.get(name);
  if (!value) throw new Error(`Missing environment variable: ${name}`);
  return value;
}

function escapeHtml(text: unknown): string {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function safeHttpUrl(value: unknown): string | null {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const url = new URL(value.trim());
    return url.protocol === "http:" || url.protocol === "https:" ? url.toString() : null;
  } catch {
    return null;
  }
}

function getSourceUrl(articleRow: Record<string, unknown> | null): string | null {
  if (!articleRow) return null;
  for (const key of [
    "url",
    "link",
    "source_url",
    "article_url",
    "original_url",
    "canonical_url",
    "href",
  ]) {
    const url = safeHttpUrl(articleRow[key]);
    if (url) return url;
  }
  return null;
}

function buildChannelPost(params: {
  title: string;
  summary: string;
  sourceUrl?: string | null;
}): string {
  const parts = [`<b>${escapeHtml(params.title)}</b>`, "", escapeHtml(params.summary)];
  if (params.sourceUrl) {
    parts.push("", `<a href="${escapeHtml(params.sourceUrl)}">Источник</a>`);
  }
  return parts.join("\n");
}

async function telegram(method: string, body: Record<string, unknown>) {
  const response = await fetch(`https://api.telegram.org/bot${BOT_TOKEN}/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) {
    throw new Error(`Telegram ${method} failed: ${JSON.stringify(result)}`);
  }
  return result;
}

async function answerCallbackQuery(
  callbackQueryId: string,
  text?: string,
  showAlert = false,
) {
  await telegram("answerCallbackQuery", {
    callback_query_id: callbackQueryId,
    text,
    show_alert: showAlert,
  });
}

async function editMessageReplyMarkup(
  chatId: number,
  messageId: number,
  inlineKeyboard: Array<Array<{ text: string; callback_data: string }>>,
) {
  await telegram("editMessageReplyMarkup", {
    chat_id: chatId,
    message_id: messageId,
    reply_markup: { inline_keyboard: inlineKeyboard },
  });
}

function candidateKeyboard(queueId: number, revision: number) {
  return [[
    { text: "✅ Опубликовать", callback_data: `publish:${queueId}:${revision}` },
    { text: "❌ Не публиковать", callback_data: `reject:${queueId}:${revision}` },
  ]];
}

function rejectionReasonKeyboard(queueId: number, revision: number) {
  return [
    [{
      text: "✍️ Исправить текст",
      callback_data: `feedback:text_correction:${queueId}:${revision}`,
    }],
    [{
      text: "🎯 Не подходит тематика",
      callback_data: `feedback:topic_mismatch:${queueId}:${revision}`,
    }],
    [{
      text: "🗂 Другая причина",
      callback_data: `feedback:other_rejection:${queueId}:${revision}`,
    }],
    [{ text: "↩️ Назад", callback_data: `back:${queueId}:${revision}` }],
  ];
}

function parsePositiveInteger(value: string | undefined): number | null {
  if (!value || !/^\d+$/.test(value)) return null;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
}

function parseCallbackData(raw: string): CallbackAction | null {
  if (raw === "done") return { kind: "done" };
  const parts = raw.split(":");

  if (parts[0] === "pf") {
    const action = parts[1] as "details" | "activate" | "reject";
    const proposalId = parsePositiveInteger(parts[2]);
    if (!["details", "activate", "reject"].includes(action) || !proposalId) {
      return null;
    }
    return { kind: "prefilter", action, proposalId };
  }

  if (["publish", "reject", "back"].includes(parts[0])) {
    const queueId = parsePositiveInteger(parts[1]);
    if (!queueId) return null;

    let revision: number | undefined;
    if (parts[2]) {
      const parsedRevision = parsePositiveInteger(parts[2]);
      if (!parsedRevision) return null;
      revision = parsedRevision;
    }
    if (parts[0] === "publish") return { kind: "publish", queueId, revision };
    if (parts[0] === "reject") return { kind: "reject", queueId, revision };
    return { kind: "back", queueId, revision };
  }

  if (parts[0] === "feedback") {
    const feedbackType = parts[1] as FeedbackType;
    const queueId = parsePositiveInteger(parts[2]);
    const revision = parsePositiveInteger(parts[3]);
    if (
      !["text_correction", "topic_mismatch", "other_rejection"].includes(feedbackType) ||
      !queueId ||
      !revision
    ) {
      return null;
    }
    return { kind: "feedback", feedbackType, queueId, revision };
  }

  return null;
}

async function handlePrefilterProposal(
  callbackId: string,
  callbackUserId: number,
  chatId: number,
  messageId: number,
  action: Extract<CallbackAction, { kind: "prefilter" }>,
) {
  const { data: proposal, error } = await supabase
    .from("prefilter_policy_proposals")
    .select("id,status,summary,rationale,policy,metrics,telegram_chat_id,telegram_message_id")
    .eq("id", action.proposalId)
    .maybeSingle();
  if (error) throw error;
  if (!proposal) {
    await answerCallbackQuery(callbackId, "Предложение не найдено", true);
    return;
  }
  if (
    Number(proposal.telegram_chat_id) !== chatId ||
    Number(proposal.telegram_message_id) !== messageId
  ) {
    await answerCallbackQuery(callbackId, "Это неактуальное сообщение", true);
    return;
  }

  if (action.action === "details") {
    const positive = Array.isArray(proposal.policy?.positive_terms)
      ? proposal.policy.positive_terms.join(", ") || "нет"
      : "нет";
    const negative = Array.isArray(proposal.policy?.negative_terms)
      ? proposal.policy.negative_terms.join(", ") || "нет"
      : "нет";
    await telegram("sendMessage", {
      chat_id: chatId,
      text: (
        "🔎 Детали prefilter #" + proposal.id + "\n\n" +
        proposal.rationale + "\n\n" +
        "Пропускать: " + positive + "\n\n" +
        "Отсекать: " + negative + "\n\n" +
        "Это только тематические сигналы; исправления языка сюда не входят."
      ).slice(0, 4000),
      reply_parameters: { message_id: messageId, allow_sending_without_reply: true },
    });
    await answerCallbackQuery(callbackId, "Детали отправлены");
    return;
  }

  if (proposal.status !== "pending") {
    await answerCallbackQuery(callbackId, "Предложение уже обработано", true);
    return;
  }
  const rpcName = action.action === "activate"
    ? "activate_prefilter_policy"
    : "reject_prefilter_policy";
  const { data: changed, error: rpcError } = await supabase.rpc(rpcName, {
    p_proposal_id: proposal.id,
    p_editor_user_id: callbackUserId,
  });
  if (rpcError) throw rpcError;
  if (!changed) {
    await answerCallbackQuery(callbackId, "Предложение уже обработано", true);
    return;
  }
  const label = action.action === "activate"
    ? "✅ Активировано"
    : "❌ Отклонено";
  await editMessageReplyMarkup(chatId, messageId, [[
    { text: label, callback_data: "done" },
  ]]);
  await answerCallbackQuery(
    callbackId,
    action.action === "activate" ? "Новый prefilter активирован" : "Предложение отклонено",
  );
}

async function getQueue(queueId: number): Promise<QueueRow | null> {
  const { data, error } = await supabase
    .from("editor_queue")
    .select("id,article_id,status,revision,telegram_chat_id,telegram_message_id")
    .eq("id", queueId)
    .maybeSingle();
  if (error) throw error;
  return data as QueueRow | null;
}

function callbackMatchesCurrentMessage(
  queue: QueueRow,
  chatId: number,
  messageId: number,
): boolean {
  if (queue.telegram_chat_id !== null && Number(queue.telegram_chat_id) !== Number(chatId)) {
    return false;
  }
  if (
    queue.telegram_message_id !== null &&
    Number(queue.telegram_message_id) !== Number(messageId)
  ) {
    return false;
  }
  return true;
}

async function loadArticleAndAnalysis(articleId: number) {
  const [articleResult, analysisResult] = await Promise.all([
    supabase.from("articles").select("*").eq("id", articleId).single(),
    supabase.from("article_analysis").select("*").eq("article_id", articleId).single(),
  ]);
  if (articleResult.error) throw articleResult.error;
  if (analysisResult.error) throw analysisResult.error;
  return {
    article: articleResult.data as Record<string, unknown>,
    analysis: analysisResult.data as Record<string, unknown>,
  };
}

async function publishCandidate(
  callbackId: string,
  callbackUserId: number,
  chatId: number,
  messageId: number,
  queue: QueueRow,
) {
  let channelPublished = false;
  const { data: claimed, error: claimError } = await supabase
    .from("editor_queue")
    .update({ status: "publishing" })
    .eq("id", queue.id)
    .eq("revision", queue.revision)
    .eq("status", "sent")
    .select("id")
    .maybeSingle();

  if (claimError) throw claimError;
  if (!claimed) {
    await answerCallbackQuery(callbackId, "Эта новость уже обработана", true);
    return;
  }

  try {
    const { article, analysis } = await loadArticleAndAnalysis(queue.article_id);
    const title = String(
      analysis.telegram_title || analysis.russian_summary || analysis.reason || "Новость",
    );
    const summary = String(
      analysis.telegram_text || analysis.russian_summary || "Текст новости недоступен.",
    );
    const sourceUrl = getSourceUrl(article);

    await telegram("sendMessage", {
      chat_id: CHANNEL_ID,
      text: buildChannelPost({ title, summary, sourceUrl }),
      parse_mode: "HTML",
      disable_web_page_preview: false,
    });
    channelPublished = true;

    const reviewedAt = new Date().toISOString();
    const { data: approved, error: approveError } = await supabase
      .from("editor_queue")
      .update({ status: "approved", reviewed_at: reviewedAt })
      .eq("id", queue.id)
      .eq("status", "publishing")
      .eq("revision", queue.revision)
      .select("id")
      .maybeSingle();
    if (approveError) throw approveError;
    if (!approved) throw new Error("Published queue item could not be finalized");

    const { error: feedbackError } = await supabase.from("editorial_feedback").insert({
      queue_id: queue.id,
      article_id: queue.article_id,
      queue_revision: queue.revision,
      feedback_type: "approved",
      status: "applied",
      telegram_chat_id: chatId,
      telegram_user_id: callbackUserId,
      source_title: article.title ?? null,
      source_summary: article.summary ?? null,
      draft_title: analysis.telegram_title ?? null,
      draft_text: analysis.telegram_text ?? null,
      submitted_at: reviewedAt,
      processed_at: reviewedAt,
    });
    if (feedbackError && feedbackError.code !== "23505") {
      console.error("Failed to record approval feedback:", feedbackError);
    }

    await editMessageReplyMarkup(chatId, messageId, [[
      { text: "✅ Опубликовано", callback_data: "done" },
    ]]);
    await answerCallbackQuery(callbackId, "Опубликовано");
  } catch (error) {
    console.error("Publish failed:", error);
    if (!channelPublished) {
      await supabase
        .from("editor_queue")
        .update({ status: "sent" })
        .eq("id", queue.id)
        .eq("status", "publishing")
        .eq("revision", queue.revision);
      await answerCallbackQuery(callbackId, "Не удалось опубликовать", true);
    } else {
      // Do not reopen the item after Telegram accepted the channel post: that
      // could publish the same news twice. The "publishing" state is recoverable
      // manually and visibly safer than a duplicate post.
      await answerCallbackQuery(
        callbackId,
        "Опубликовано, но запись статуса требует проверки",
        true,
      );
    }
  }
}

async function showRejectionReasons(
  callbackId: string,
  chatId: number,
  messageId: number,
  queue: QueueRow,
) {
  const { data, error } = await supabase
    .from("editor_queue")
    .update({ status: "awaiting_feedback" })
    .eq("id", queue.id)
    .eq("revision", queue.revision)
    .eq("status", "sent")
    .select("id")
    .maybeSingle();
  if (error) throw error;
  if (!data) {
    await answerCallbackQuery(callbackId, "Эта новость уже обработана", true);
    return;
  }

  await editMessageReplyMarkup(
    chatId,
    messageId,
    rejectionReasonKeyboard(queue.id, queue.revision),
  );
  await answerCallbackQuery(callbackId, "Выберите причину");
}

async function returnToCandidate(
  callbackId: string,
  chatId: number,
  messageId: number,
  queue: QueueRow,
) {
  await supabase
    .from("editorial_feedback")
    .update({ status: "cancelled", updated_at: new Date().toISOString() })
    .eq("queue_id", queue.id)
    .eq("queue_revision", queue.revision)
    .eq("status", "awaiting_comment");

  const { data, error } = await supabase
    .from("editor_queue")
    .update({ status: "sent" })
    .eq("id", queue.id)
    .eq("revision", queue.revision)
    .eq("status", "awaiting_feedback")
    .select("id")
    .maybeSingle();
  if (error) throw error;
  if (!data) {
    await answerCallbackQuery(callbackId, "Состояние новости уже изменилось", true);
    return;
  }

  await editMessageReplyMarkup(chatId, messageId, candidateKeyboard(queue.id, queue.revision));
  await answerCallbackQuery(callbackId, "Отмена");
}

function feedbackPrompt(type: FeedbackType): { text: string; placeholder: string } {
  if (type === "text_correction") {
    return {
      text: "Что именно нужно исправить в описании или русском тексте? Ответьте на это сообщение одним комментарием.",
      placeholder: "Например: неверно указана дата…",
    };
  }
  if (type === "topic_mismatch") {
    return {
      text: "Почему эта тема не подходит каналу? Ответьте на это сообщение — причина будет учтена при будущем отборе.",
      placeholder: "Например: слишком локальная политика…",
    };
  }
  return {
    text: "Почему новость не должна публиковаться? Ответьте на это сообщение одним комментарием.",
    placeholder: "Например: устарела или повторяет другую…",
  };
}

async function requestFeedbackComment(
  callbackId: string,
  callbackUserId: number,
  chatId: number,
  messageId: number,
  queue: QueueRow,
  feedbackType: FeedbackType,
) {
  if (queue.status !== "awaiting_feedback") {
    await answerCallbackQuery(callbackId, "Состояние новости уже изменилось", true);
    return;
  }

  const { data: existing, error: existingError } = await supabase
    .from("editorial_feedback")
    .select("id,prompt_message_id,telegram_user_id")
    .eq("queue_id", queue.id)
    .eq("queue_revision", queue.revision)
    .eq("status", "awaiting_comment")
    .maybeSingle();
  if (existingError) throw existingError;
  if (existing) {
    await answerCallbackQuery(callbackId, "Бот уже ждёт комментарий", true);
    return;
  }

  const { article, analysis } = await loadArticleAndAnalysis(queue.article_id);
  const { data: feedback, error: insertError } = await supabase
    .from("editorial_feedback")
    .insert({
      queue_id: queue.id,
      article_id: queue.article_id,
      queue_revision: queue.revision,
      feedback_type: feedbackType,
      status: "awaiting_comment",
      telegram_chat_id: chatId,
      telegram_user_id: callbackUserId,
      source_title: article.title ?? null,
      source_summary: article.summary ?? null,
      draft_title: analysis.telegram_title ?? null,
      draft_text: analysis.telegram_text ?? null,
    })
    .select("id")
    .single();
  if (insertError) throw insertError;

  try {
    const prompt = feedbackPrompt(feedbackType);
    const promptResult = await telegram("sendMessage", {
      chat_id: chatId,
      text: prompt.text,
      reply_parameters: { message_id: messageId, allow_sending_without_reply: true },
      reply_markup: {
        force_reply: true,
        selective: true,
        input_field_placeholder: prompt.placeholder,
      },
    });

    const promptMessageId = Number(promptResult.result?.message_id);
    if (!Number.isSafeInteger(promptMessageId) || promptMessageId <= 0) {
      throw new Error("Telegram feedback prompt has no valid message_id");
    }
    const { error: promptUpdateError } = await supabase
      .from("editorial_feedback")
      .update({ prompt_message_id: promptMessageId, updated_at: new Date().toISOString() })
      .eq("id", feedback.id);
    if (promptUpdateError) throw promptUpdateError;

    await editMessageReplyMarkup(chatId, messageId, [[
      { text: "⏳ Жду комментарий", callback_data: "done" },
      { text: "↩️ Отмена", callback_data: `back:${queue.id}:${queue.revision}` },
    ]]);
    await answerCallbackQuery(callbackId, "Напишите комментарий ответом боту");
  } catch (error) {
    await supabase
      .from("editorial_feedback")
      .update({ status: "cancelled", error: String(error), updated_at: new Date().toISOString() })
      .eq("id", feedback.id);
    throw error;
  }
}

async function handleCallback(callback: Record<string, any>) {
  const callbackId = String(callback.id || "");
  const callbackUserId = Number(callback.from?.id);
  const chatId = Number(callback.message?.chat?.id);
  const messageId = Number(callback.message?.message_id);
  const action = parseCallbackData(String(callback.data || ""));

  if (!action || !callbackId || !callbackUserId || !chatId || !messageId) {
    if (callbackId) await answerCallbackQuery(callbackId, "Некорректная команда", true);
    return;
  }
  if (action.kind === "done") {
    await answerCallbackQuery(callbackId);
    return;
  }
  if (action.kind === "prefilter") {
    await handlePrefilterProposal(
      callbackId,
      callbackUserId,
      chatId,
      messageId,
      action,
    );
    return;
  }

  const queue = await getQueue(action.queueId);
  if (!queue) {
    await answerCallbackQuery(callbackId, "Новость не найдена", true);
    return;
  }
  if (!callbackMatchesCurrentMessage(queue, chatId, messageId)) {
    await answerCallbackQuery(callbackId, "Это неактуальное сообщение", true);
    return;
  }

  const expectedRevision = action.revision ?? queue.revision;
  if (expectedRevision !== queue.revision) {
    await answerCallbackQuery(callbackId, "Это старая версия новости", true);
    return;
  }

  if (action.kind === "publish") {
    if (queue.status !== "sent") {
      await answerCallbackQuery(callbackId, "Эта новость уже обработана", true);
      return;
    }
    await publishCandidate(callbackId, callbackUserId, chatId, messageId, queue);
  } else if (action.kind === "reject") {
    await showRejectionReasons(callbackId, chatId, messageId, queue);
  } else if (action.kind === "back") {
    await returnToCandidate(callbackId, chatId, messageId, queue);
  } else if (action.kind === "feedback") {
    await requestFeedbackComment(
      callbackId,
      callbackUserId,
      chatId,
      messageId,
      queue,
      action.feedbackType,
    );
  }
}

async function handleEditorComment(message: Record<string, any>) {
  const text = String(message.text || "").trim();
  const chatId = Number(message.chat?.id);
  const userId = Number(message.from?.id);
  const replyToMessageId = Number(message.reply_to_message?.message_id);
  if (!text || !chatId || !userId || !replyToMessageId) return;

  const { data: feedback, error } = await supabase
    .from("editorial_feedback")
    .select("id,queue_id,article_id,queue_revision,feedback_type")
    .eq("status", "awaiting_comment")
    .eq("telegram_chat_id", chatId)
    .eq("telegram_user_id", userId)
    .eq("prompt_message_id", replyToMessageId)
    .maybeSingle();
  if (error) throw error;
  if (!feedback) return;

  const { data: result, error: submitError } = await supabase.rpc(
    "submit_editorial_feedback",
    { p_feedback_id: feedback.id, p_editor_comment: text },
  );
  if (submitError) throw submitError;

  let correctionDispatchStatus = "";
  if (result === "correction_pending") {
    const { data: dispatchStatus, error: dispatchError } = await supabase.rpc(
      "dispatch_editorial_correction",
      { p_feedback_id: feedback.id },
    );
    if (dispatchError) {
      // The correction remains pending and will be recovered by the regular
      // analyzer. Do not turn a saved editor comment into a failed webhook.
      console.error("Immediate correction dispatch failed:", dispatchError);
    } else {
      correctionDispatchStatus = String(dispatchStatus || "");
    }
  }

  const queue = await getQueue(Number(feedback.queue_id));
  if (queue?.telegram_chat_id && queue.telegram_message_id) {
    const statusText = result === "correction_pending"
      ? correctionDispatchStatus === "queued"
        ? "🛠 Исправление запущено"
        : "⏳ Исправление в очереди"
      : feedback.feedback_type === "topic_mismatch"
      ? "🎯 Отклонено: тематика"
      : "❌ Отклонено";
    await editMessageReplyMarkup(
      Number(queue.telegram_chat_id),
      Number(queue.telegram_message_id),
      [[{ text: statusText, callback_data: "done" }]],
    );
  }

  const acknowledgement = result === "correction_pending"
    ? correctionDispatchStatus === "queued"
      ? "Принято. ИИ исправляет текст по вашему комментарию. Исправленная версия придёт сюда сразу для проверки."
      : "Принято. Комментарий сохранён. Немедленный запуск пока недоступен; исправление будет обработано ближайшим полным запуском."
    : feedback.feedback_type === "topic_mismatch"
    ? "Принято. Новость отклонена, а причина будет учтена при будущем тематическом отборе."
    : "Принято. Новость отклонена; комментарий сохранён, но не меняет тематическую политику.";

  await telegram("sendMessage", {
    chat_id: chatId,
    text: acknowledgement,
    reply_parameters: { message_id: message.message_id, allow_sending_without_reply: true },
  });
}

Deno.serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("method not allowed", { status: 405 });
  }

  try {
    const payload = await req.json();
    if (payload.callback_query) {
      await handleCallback(payload.callback_query);
    } else if (payload.message) {
      await handleEditorComment(payload.message);
    }
    return new Response("ok", { status: 200 });
  } catch (error) {
    console.error("Webhook error:", error);
    return new Response("internal error", { status: 500 });
  }
});
