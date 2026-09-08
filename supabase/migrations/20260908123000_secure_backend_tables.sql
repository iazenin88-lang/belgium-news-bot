-- Keep backend-only news pipeline tables inaccessible through public API roles.
begin;

alter table public.sources enable row level security;
alter table public.articles enable row level security;
alter table public.article_analysis enable row level security;
alter table public.editor_queue enable row level security;
alter table public.ai_runs enable row level security;
alter table public.ai_balance enable row level security;

revoke all privileges on table
  public.sources,
  public.articles,
  public.article_analysis,
  public.editor_queue,
  public.ai_runs,
  public.ai_balance
from anon, authenticated;

-- Identity/serial sequences are a separate privilege surface in PostgreSQL.
revoke all privileges on sequence
  public.sources_id_seq,
  public.articles_id_seq,
  public.article_analysis_id_seq,
  public.editor_queue_id_seq,
  public.ai_runs_id_seq
from anon, authenticated;

-- The collector, analyzer, notifier, balance reporter, and Telegram webhook use
-- the service role. Preserve its existing backend access explicitly.
grant all privileges on table
  public.sources,
  public.articles,
  public.article_analysis,
  public.editor_queue,
  public.ai_runs,
  public.ai_balance
to service_role;

grant usage, select on sequence
  public.sources_id_seq,
  public.articles_id_seq,
  public.article_analysis_id_seq,
  public.editor_queue_id_seq,
  public.ai_runs_id_seq
to service_role;

commit;
