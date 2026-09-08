drop index if exists public.editorial_feedback_policy_history;

create index editorial_feedback_policy_history
  on public.editorial_feedback (created_at desc, id desc)
  where status = 'applied'
    and feedback_type in ('approved', 'topic_mismatch', 'text_correction');
