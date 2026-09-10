-- Store the model's cross-source event decision alongside the article analysis.
-- The table is already backend-only (RLS + service_role grants from the
-- security hardening migration), so these fields do not expose article data.
begin;

alter table public.article_analysis
  add column if not exists is_duplicate_event boolean not null default false,
  add column if not exists duplicate_of_article_id bigint
    references public.articles(id) on delete set null,
  add column if not exists duplicate_reason text,
  add column if not exists is_material_update boolean not null default false;

alter table public.article_analysis
  add constraint article_analysis_material_update_requires_duplicate
  check (is_material_update = false or is_duplicate_event = true);

create index if not exists article_analysis_duplicate_of_idx
  on public.article_analysis (duplicate_of_article_id)
  where duplicate_of_article_id is not null;

create index if not exists editor_queue_event_coverage_idx
  on public.editor_queue (created_at desc, status)
  where status in (
    'pending', 'notifying', 'sent', 'awaiting_feedback',
    'correction_pending', 'publishing', 'approved', 'published'
  );

comment on column public.article_analysis.is_duplicate_event is
  'AI decision that this article describes an already covered real-world event.';
comment on column public.article_analysis.duplicate_of_article_id is
  'Article from recent coverage history that describes the same event.';
comment on column public.article_analysis.duplicate_reason is
  'Short explanation for the cross-source duplicate decision.';
comment on column public.article_analysis.is_material_update is
  'True when the same event has a substantial new development worth proposing.';

commit;
