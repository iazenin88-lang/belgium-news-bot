-- Roll back semantic cross-source deduplication metadata.
-- Run only after deploying the previous analyzer version and confirming no
-- worker still writes the four columns below.
begin;

drop index if exists public.article_analysis_duplicate_of_idx;
drop index if exists public.editor_queue_event_coverage_idx;

alter table public.article_analysis
  drop constraint if exists article_analysis_material_update_requires_duplicate,
  drop column if exists is_material_update,
  drop column if exists duplicate_reason,
  drop column if exists duplicate_of_article_id,
  drop column if exists is_duplicate_event;

commit;
