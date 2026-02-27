"""
bot.py – Telegram bot entry-point.

Features
────────
• Send a text message  → answered by the configured AI
• Send one or more photos (with optional caption) → AI sees images + text
• Send a PDF document → text is extracted and stored as context; AI is asked
  immediately if a caption was included, otherwise the bot confirms receipt
• /start   – greet user
• /clear   – wipe conversation history
• /help    – show usage
• /model   – show current provider & model

Per-user conversation history is kept in memory (resets on restart).
"""

import asyncio
import logging
import mimetypes
import re
from collections import defaultdict

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from ai_handler import ask_ai, encode_image_bytes, make_pdf_part, make_text_part
from pdf_handler import extract_pdf_text

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# user_id -> list of history dicts
user_history: dict[int, list[dict]] = defaultdict(list)

# Temporary buffer for media group batching:
# media_group_id -> {"parts": [...], "timer": asyncio.Task, "chat_id": ..., "user_id": ...}
_media_groups: dict[str, dict] = {}
_MEDIA_GROUP_TIMEOUT = 1.5  # seconds to wait before processing a media group


# ── helpers ──────────────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove markdown syntax characters, keeping plain readable text."""
    # Remove headers (### Heading → Heading)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove bold/italic (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.+?)_{1,3}', r'\1', text)
    # Remove inline code (`code`)
    text = re.sub(r'`{1,3}(.+?)`{1,3}', r'\1', text, flags=re.DOTALL)
    # Remove horizontal rules (--- or ***)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Remove blockquotes (> text)
    text = re.sub(r'^>+\s?', '', text, flags=re.MULTILINE)
    # Remove markdown links [text](url) → text
    text = re.sub(r'\[(.+?)\]\(.*?\)', r'\1', text)
    # Remove images ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Clean up extra blank lines (3+ → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _trim_history(uid: int) -> None:
    """Keep history within MAX_HISTORY pairs (user + assistant = 2 items)."""
    hist = user_history[uid]
    max_items = config.MAX_HISTORY * 2
    if len(hist) > max_items:
        user_history[uid] = hist[-max_items:]


async def _get_image_bytes(update: Update, context: ContextTypes.DEFAULT_TYPE,
                           photo_sizes) -> bytes:
    """Download the highest-resolution version of a Telegram photo."""
    file = await context.bot.get_file(photo_sizes[-1].file_id)
    return await file.download_as_bytearray()


async def _reply_thinking(update: Update) -> None:
    await update.effective_chat.send_action("typing")


async def _send_ai_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    uid: int,
    user_parts,           # str or list[dict]
) -> None:
    """Append user turn, call AI, append assistant turn, reply."""
    user_history[uid].append({"role": "user", "content": user_parts})
    _trim_history(uid)

    await _reply_thinking(update)
    try:
        reply = ask_ai(user_history[uid])
    except Exception as exc:
        logger.exception("AI error")
        reply = f"⚠️ AI error: {exc}"
        user_history[uid].pop()  # remove failed turn
        await update.effective_message.reply_text(reply)
        return

    user_history[uid].append({"role": "assistant", "content": reply})
    _trim_history(uid)

    clean_reply = _strip_markdown(reply)
    # Telegram message limit is 4096 chars; split if needed
    for chunk in _split(clean_reply, 4096):
        await update.effective_message.reply_text(chunk)


def _split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, max(len(text), 1), size)]


# ── command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = update.effective_user.first_name or "there"
    await update.message.reply_text(
        f"👋 Hello, {name}!\n\n"
        "I'm your AI assistant. You can:\n"
        "• Ask me any question\n"
        "• Send a photo (with an optional question as caption)\n"
        "• Send a PDF (with an optional question as caption)\n"
        "• Send multiple photos at once\n\n"
        "Use /help for more info or /clear to reset the conversation."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model_name = config.get_model()
    provider = config.AI_PROVIDER
    await update.message.reply_text(
        f"*AI Assistant Bot*\n\n"
        f"Provider: `{provider}`\n"
        f"Model: `{model_name}`\n\n"
        "*Commands*\n"
        "/start – welcome message\n"
        "/clear – erase conversation history\n"
        "/help  – this message\n"
        "/model – show current AI model\n\n"
        "*Usage*\n"
        "• Text → question answered by AI\n"
        "• Photo → AI analyses the image; add a caption for a specific question\n"
        "• PDF   → text extracted; AI answers if you add a caption\n"
        "• Multiple photos → send as an album; add a caption to the last one\n\n"
        "Conversation history is preserved across messages (use /clear to reset).",
        parse_mode="Markdown",
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    user_history[uid].clear()
    await update.message.reply_text("🗑️ Conversation history cleared.")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Provider: `{config.AI_PROVIDER}`\nModel: `{config.get_model()}`",
        parse_mode="Markdown",
    )


# ── message handlers ──────────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    text = update.message.text.strip()
    if not text:
        return
    await _send_ai_reply(update, context, uid, text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    message = update.message
    caption = (message.caption or "").strip()

    media_group_id = message.media_group_id

    if media_group_id:
        # Buffer into media group so we can collect all images before sending
        img_bytes = await _get_image_bytes(update, context, message.photo)
        part = encode_image_bytes(bytes(img_bytes), "image/jpeg")

        if media_group_id not in _media_groups:
            _media_groups[media_group_id] = {
                "parts": [],
                "caption": "",
                "update": update,
                "context": context,
                "uid": uid,
                "task": None,
            }

        _media_groups[media_group_id]["parts"].append(part)
        if caption:
            _media_groups[media_group_id]["caption"] = caption

        # Cancel previous timer and start a new one
        existing = _media_groups[media_group_id]["task"]
        if existing:
            existing.cancel()
        _media_groups[media_group_id]["task"] = asyncio.get_event_loop().call_later(
            _MEDIA_GROUP_TIMEOUT,
            lambda mgid=media_group_id: asyncio.ensure_future(
                _flush_media_group(mgid)
            ),
        )
    else:
        # Single photo
        img_bytes = await _get_image_bytes(update, context, message.photo)
        parts = [encode_image_bytes(bytes(img_bytes), "image/jpeg")]
        question = caption or "Please describe what you see in this image."
        parts.append(make_text_part(question))
        await _send_ai_reply(update, context, uid, parts)


async def _flush_media_group(media_group_id: str) -> None:
    group = _media_groups.pop(media_group_id, None)
    if not group:
        return
    parts = group["parts"]
    caption = group["caption"] or "Please describe what you see in these images."
    update = group["update"]
    context = group["context"]
    uid = group["uid"]

    parts.append(make_text_part(caption))
    await _send_ai_reply(update, context, uid, parts)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    doc = update.message.document
    caption = (update.message.caption or "").strip()
    mime = doc.mime_type or ""

    file = await context.bot.get_file(doc.file_id)
    data = bytes(await file.download_as_bytearray())

    # ── Image sent as file ────────────────────────────────────────────────────
    if mime.startswith("image/"):
        parts = [encode_image_bytes(data, mime)]
        question = caption or "Please describe what you see in this image."
        parts.append(make_text_part(question))
        await _send_ai_reply(update, context, uid, parts)
        return

    # ── PDF ───────────────────────────────────────────────────────────────────
    if mime == "application/pdf" or doc.file_name.lower().endswith(".pdf"):
        await update.effective_chat.send_action("typing")
        pdf_text = extract_pdf_text(data)
        parts = [make_pdf_part(pdf_text)]

        if caption:
            parts.append(make_text_part(caption))
            await _send_ai_reply(update, context, uid, parts)
        else:
            # Store PDF context without triggering a reply
            user_history[uid].append({"role": "user", "content": parts})
            _trim_history(uid)
            await update.message.reply_text(
                "📄 PDF received and stored as context.\n"
                "Now ask me anything about it!"
            )
        return

    # ── Unsupported ───────────────────────────────────────────────────────────
    await update.message.reply_text(
        f"⚠️ Unsupported file type: `{mime or doc.file_name}`\n"
        "Please send an image or a PDF.",
        parse_mode="Markdown",
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    if not config.AI_API_KEY:
        raise RuntimeError("AI_API_KEY is not set in .env")

    app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Bot started. Provider=%s  Model=%s", config.AI_PROVIDER, config.get_model())
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
