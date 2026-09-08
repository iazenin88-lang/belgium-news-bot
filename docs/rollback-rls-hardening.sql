-- Emergency rollback for 20260908123000_secure_backend_tables.sql.
-- This restores the exact public table/sequence access observed before Backup 04.
begin;

alter table public.sources disable row level security;
alter table public.articles disable row level security;
alter table public.article_analysis disable row level security;
alter table public.editor_queue disable row level security;
alter table public.ai_runs disable row level security;
alter table public.ai_balance disable row level security;

grant all privileges on table
  public.sources,
  public.articles,
  public.article_analysis,
  public.editor_queue,
  public.ai_runs,
  public.ai_balance
to anon, authenticated;

grant usage, select on sequence
  public.sources_id_seq,
  public.articles_id_seq,
  public.article_analysis_id_seq,
  public.editor_queue_id_seq,
  public.ai_runs_id_seq
to anon, authenticated;

commit;
