-- Persistent semantic memory for every applied editorial relevance decision.
begin;

create extension if not exists vector
with schema extensions;

alter table public.articles
  add column if not exists relevance_embedding extensions.vector(512),
  add column if not exists relevance_embedding_model text,
  add column if not exists relevance_embedding_hash text,
  add column if not exists relevance_embedding_updated_at timestamptz;

comment on column public.articles.relevance_embedding is
  'Semantic representation used to retrieve similar approved/topic-rejected articles.';
comment on column public.articles.relevance_embedding_model is
  'Embedding profile. Comparisons are valid only inside the same profile.';
comment on column public.articles.relevance_embedding_hash is
  'SHA-256 of the normalized text used to create relevance_embedding.';
comment on column public.articles.relevance_embedding_updated_at is
  'Last successful relevance embedding generation time.';

create or replace function public.match_editorial_decisions(
  p_query_embedding extensions.vector(512),
  p_embedding_model text,
  p_match_count_per_type integer default 5,
  p_min_similarity double precision default 0.30
)
returns table (
  feedback_id bigint,
  article_id bigint,
  feedback_type text,
  status text,
  editor_comment text,
  source_title text,
  source_summary text,
  draft_title text,
  draft_text text,
  created_at timestamptz,
  similarity double precision
)
language sql
stable
security invoker
set search_path = ''
as $$
  with candidates as (
    select
      f.id as feedback_id,
      f.article_id,
      f.feedback_type,
      f.status,
      f.editor_comment,
      f.source_title,
      f.source_summary,
      f.draft_title,
      f.draft_text,
      f.created_at,
      1 - (
        a.relevance_embedding operator(extensions.<=>) p_query_embedding
      ) as similarity,
      row_number() over (
        partition by f.feedback_type
        order by
          a.relevance_embedding operator(extensions.<=>) p_query_embedding,
          f.created_at desc,
          f.id desc
      ) as label_rank
    from public.editorial_feedback as f
    join public.articles as a on a.id = f.article_id
    where f.status = 'applied'
      and f.feedback_type in ('approved', 'topic_mismatch')
      and a.relevance_embedding is not null
      and a.relevance_embedding_model = p_embedding_model
      and 1 - (
        a.relevance_embedding operator(extensions.<=>) p_query_embedding
      ) >= p_min_similarity
  )
  select
    candidates.feedback_id,
    candidates.article_id,
    candidates.feedback_type,
    candidates.status,
    candidates.editor_comment,
    candidates.source_title,
    candidates.source_summary,
    candidates.draft_title,
    candidates.draft_text,
    candidates.created_at,
    candidates.similarity
  from candidates
  where candidates.label_rank <= least(greatest(p_match_count_per_type, 1), 10)
  order by candidates.similarity desc, candidates.created_at desc;
$$;

revoke all on function public.match_editorial_decisions(
  extensions.vector(512), text, integer, double precision
) from public, anon, authenticated;
grant execute on function public.match_editorial_decisions(
  extensions.vector(512), text, integer, double precision
) to service_role;

commit;
